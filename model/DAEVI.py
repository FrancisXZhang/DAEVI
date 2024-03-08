''' Spatial-Temporal Transformer Networks
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.spectral_norm import spectral_norm as _spectral_norm

import logging

# Configure logging with time format
logging.basicConfig(
    filename= 'SF.log',
    filemode='w',
    format='%(name)s - %(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

# set the logging level
logging.getLogger().setLevel(logging.DEBUG)

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).' % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


class InpaintGenerator(BaseNetwork):
    def __init__(self, init_weights=True, in_channels=3):
        super(InpaintGenerator, self).__init__()
        channel = 256
        stack_num = 4
        #patchsize = [(108, 60), (36, 20), (18, 10), (9, 5)] #removed by rema
        patchsize = [(72, 72), (24, 24), (12, 12), (6, 6)]
        blocks = []
        for _ in range(stack_num):
            blocks.append(TransformerBlock(patchsize, hidden=channel))
        self.transformer = nn.Sequential(*blocks)

        blocks = []
        for _ in range(stack_num):
            blocks.append(DepthTransformerBlock(patchsize, hidden=channel))
        self.transformer_depth = nn.Sequential(*blocks)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # decoder: decode frames from features
        self.decoder = nn.Sequential(
            deconv(channel, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        )

        # decoder: decode depths from features
        self.decoder_depth = nn.Sequential(
            deconv(channel, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        )

        if init_weights:
            self.init_weights()

    def forward(self, masked_frames, masks):
        # extracting features
        b, t, c, h, w = masked_frames.size()
        masks = masks.view(b*t, 1, h, w)
        enc_feat = self.encoder(masked_frames.view(b*t, c, h, w))
        _, c, h, w = enc_feat.size()

        # transformer
        masks = F.interpolate(masks, scale_factor=1.0/4)
        output = self.transformer(
            {'x': enc_feat, 'm': masks, 'b': b, 'c': c, 'd': enc_feat})
        enc_feat = output['x']
        d_feat = output['d']

        # depth estimation
        depths = self.decoder_depth(d_feat)

        # transformer for depth
        output = self.transformer_depth(
            {'x': enc_feat, 'm': masks, 'b': b, 'c': c, 'd': depths})

        # decoder
        output = self.decoder(output['x'])
        output = torch.tanh(output)

        return output, depths


    def infer(self, enc_feat, masks):
        t, c, h, w = masks.size()
        masks = masks.view(t, c, h, w)
        
        b, c, h, w = enc_feat.size()
        # transformer
        masks = F.interpolate(masks, scale_factor=1.0/4)
        output = self.transformer(
            {'x': enc_feat, 'm': masks, 'b': b, 'c': c, 'd': enc_feat})
        enc_feat = output['x']
        d_feat = output['d']


        # depth estimation
        depths = self.decoder_depth(d_feat)

        # transformer for depth
        output = self.transformer_depth(
            {'x': enc_feat, 'm': masks, 'b': b, 'c': c, 'd': depths})

        enc_feat = output['x']
        
        return enc_feat, depths

class deconv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel,
                              kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)
        return self.conv(x)


# #############################################################################
# ############################# Transformer  ##################################
# #############################################################################
class InterleavedGroupConv(nn.Module):
    def __init__(self, num_features, num_groups):
        super(InterleavedGroupConv, self).__init__()
        # Assuming num_features is divisible by num_groups for simplicity
        self.group_conv = nn.Conv2d(num_features + num_groups, num_features, kernel_size=3, groups=num_groups, padding=1)

    def forward(self, features, depth):
        # Expand the depth map to have the same number of channels as features
        b, _, h, w = features.size()
        depth_expanded = depth.repeat(1, features.size(1), 1, 1)

        # Interleave the features and depth maps
        batch_size, channels, height, width = features.size()
        features_with_depth = torch.zeros(batch_size, channels * 2, height, width, device=features.device)

        # Assign features and depth maps to the interleaved tensor
        features_with_depth[:, ::2] = features
        features_with_depth[:, 1::2] = depth_expanded

        

        # Apply the group-wise convolution
        features_after_depth = self.group_conv(features_with_depth)
        return features_after_depth

# #############################################################################
# ############################# Transformer  ##################################
# #############################################################################

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, m):
        scores = torch.matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(query.size(-1))
        scores.masked_fill(m, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_val = torch.matmul(p_attn, value)
        return p_val, p_attn

class DepthAttention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention' with depth information and channel attention.
    """

    def __init__(self):
        super(DepthAttention, self).__init__()



    def forward(self, query, key, value, mask, depth):

        # Compute attention scores using the updated query
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        scores = scores * torch.sigmoid(depth)
        scores.masked_fill_(mask, float('-inf'))
        
        # Compute attention probabilities
        p_attn = F.softmax(scores, dim=-1)
        
        # Apply attention to value vectors
        p_val = torch.matmul(p_attn, value)

        return p_val, p_attn

class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads, modified to incorporate depth information.
    """

    def __init__(self, patchsize, d_model):
        super().__init__()
        self.patchsize = patchsize
        self.query_embedding = nn.Conv2d(d_model, d_model, kernel_size=1, padding=0)
        self.key_embedding = nn.Conv2d(d_model, d_model, kernel_size=1, padding=0)
        self.value_embedding = nn.Conv2d(d_model, d_model, kernel_size=1, padding=0)
        self.output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))
        self.dep_output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))
        
        self.attention = Attention()
        self.value_embedding_le = nn.Conv2d(
            d_model, d_model, kernel_size=3, padding=1, groups = d_model)
        

    def forward(self, x, m, b, c, depth_map):

        bt, _, h, w = x.size()
        t = bt // b
        d_k = c // len(self.patchsize)
        output = []
        
        _query = self.query_embedding(x) 
        _key = self.key_embedding(x) 
        _value = self.value_embedding(x) + self.value_embedding_le(x)

        for (width, height), query, key, value in zip(self.patchsize,
                                                      torch.chunk(_query, len(self.patchsize), dim=1), 
                                                      torch.chunk(_key, len(self.patchsize), dim=1),
                                                      torch.chunk(_value, len(self.patchsize), dim=1),
                                                      ):
            out_w, out_h = w // width, h // height
            mm = m.view(b, t, 1, out_h, height, out_w, width)
            mm = mm.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(b,  t*out_h*out_w, height*width)
            mm = (mm.mean(-1) > 0.5).unsqueeze(1).repeat(1, t*out_h*out_w, 1)
            
            # 1) embedding and reshape
            query = query.view(b, t, d_k, out_h, height, out_w, width)
            query = query.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)
            key = key.view(b, t, d_k, out_h, height, out_w, width)
            key = key.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)
            value = value.view(b, t, d_k, out_h, height, out_w, width)
            value = value.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)
            '''
            # 2) Apply attention on all the projected vectors in batch.
            tmp1 = []
            for q,k,v in zip(torch.chunk(query, b, dim=0), torch.chunk(key, b, dim=0), torch.chunk(value, b, dim=0)):
                y, _ = self.attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
                tmp1.append(y)
            y = torch.cat(tmp1,1)
            '''
            y, _ = self.attention(query, key, value, mm)
            # 3) "Concat" using a view and apply a final linear.
            y = y.view(b, t, out_h, out_w, d_k, height, width)
            y = y.permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(bt, d_k, h, w)
            output.append(y)
        output = torch.cat(output, 1)
        x = self.output_linear(output)
        dep_output = self.dep_output_linear(output)
        return x, dep_output


class MultiHeadedDepthAttention(nn.Module):
    """
    Take in model size and number of heads, modified to incorporate depth information.
    """

    def __init__(self, patchsize, d_model):
        super().__init__()
        self.patchsize = patchsize
        self.query_embedding = nn.Conv2d(d_model, d_model, kernel_size=1, padding=0)
        self.key_embedding = nn.Conv2d(d_model, d_model, kernel_size=1, padding=0)
        self.value_embedding = nn.Conv2d(d_model, d_model, kernel_size=1, padding=0)
        self.output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))
        
        # Depth encoding
        self.depth_enc = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride = 2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, 1, kernel_size=3, stride = 2,padding=1),
            nn.ReLU(inplace=True))

        
        self.min_attention = DepthAttention()        
        self.median_attention = DepthAttention()
        self.max_attention = DepthAttention()
        self.value_embedding_le = nn.Conv2d(
            d_model, d_model, kernel_size=3, padding=1, groups = d_model)
        

    def forward(self, x, m, b, c, depth_map):

        bt, _, h, w = x.size()
        t = bt // b
        d_k = c // len(self.patchsize)
        output = []
        
        _query = self.query_embedding(x) 
        _key = self.key_embedding(x) 
        _value = self.value_embedding(x) + self.value_embedding_le(x)
        _depth = self.depth_enc(depth_map)

        for (width, height), query, key, value in zip(self.patchsize,
                                                      torch.chunk(_query, len(self.patchsize), dim=1), 
                                                      torch.chunk(_key, len(self.patchsize), dim=1),
                                                      torch.chunk(_value, len(self.patchsize), dim=1),
                                                      ):
            out_w, out_h = w // width, h // height
            mm = m.view(b, t, 1, out_h, height, out_w, width)
            mm = mm.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(b,  t*out_h*out_w, height*width)
            mm = (mm.mean(-1) > 0.5).unsqueeze(1).repeat(1, t*out_h*out_w, 1)

            depth = _depth.view(b, t, 1, out_h, height, out_w, width)
            depth = depth.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, height*width)
            min_depth = (depth.min(-1).values).unsqueeze(1).repeat(1, t*out_h*out_w, 1)
            mean_depth = (depth.median(-1).values).unsqueeze(1).repeat(1, t*out_h*out_w, 1)
            max_depth = (depth.max(-1).values).unsqueeze(1).repeat(1, t*out_h*out_w, 1)
            
            # 1) embedding and reshape
            query = query.view(b, t, d_k, out_h, height, out_w, width)
            query = query.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)
            key = key.view(b, t, d_k, out_h, height, out_w, width)
            key = key.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)
            value = value.view(b, t, d_k, out_h, height, out_w, width)
            value = value.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)
            '''
            # 2) Apply attention on all the projected vectors in batch.
            tmp1 = []
            for q,k,v in zip(torch.chunk(query, b, dim=0), torch.chunk(key, b, dim=0), torch.chunk(value, b, dim=0)):
                y, _ = self.attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
                tmp1.append(y)
            y = torch.cat(tmp1,1)
            '''
            y, _ = self.median_attention(query, key, value, mm, mean_depth)
            y_max, _ = self.max_attention(query, key, value, mm, max_depth)
            y_min, _ = self.min_attention(query, key, value, mm, min_depth)
            y = (y + y_max + y_min)/3           
            # 3) "Concat" using a view and apply a final linear.
            y = y.view(b, t, out_h, out_w, d_k, height, width)
            y = y.permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(bt, d_k, h, w)
            output.append(y)
        output = torch.cat(output, 1)
        x = self.output_linear(output)
        return x, depth_map


# Standard 2 layerd FFN of transformer
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super(FeedForward, self).__init__()
        # We set d_ff as a default to 2048
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class TransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, patchsize, hidden=128):
        super().__init__()
        self.attention = MultiHeadedAttention(patchsize, d_model=hidden)
        self.feed_forward = FeedForward(hidden)

    def forward(self, x):
        x, m, b, c, d = x['x'], x['m'], x['b'], x['c'], x['d']
        res, res_d = self.attention(x, m, b, c, d)
        x = x + res
        x = x + self.feed_forward(x)
        d = d + res_d
        return {'x': x, 'm': m, 'b': b, 'c': c, 'd': d}


class DepthTransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, patchsize, hidden=128):
        super().__init__()
        self.attention = MultiHeadedDepthAttention(patchsize, d_model=hidden)
        self.feed_forward = FeedForward(hidden)

    def forward(self, x):
        x, m, b, c, d = x['x'], x['m'], x['b'], x['c'], x['d']
        res, _ = self.attention(x, m, b, c, d)
        x = x + res
        x = x + self.feed_forward(x)
        return {'x': x, 'm': m, 'b': b, 'c': c, 'd': d}

# ######################################################################
# ######################################################################


class Discriminator(BaseNetwork):
    def __init__(self, in_channels=3, use_sigmoid=False, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        nf = 64

        self.conv = nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels=in_channels, out_channels=nf*1, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=1, bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(64, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf*1, nf*2, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 2, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5),
                      stride=(1, 2, 2), padding=(1, 2, 2))
        )

        if init_weights:
            self.init_weights()

    def forward(self, xs):
        # B, T, C, H, W = xs.shape
        xs_t = torch.transpose(xs, 1, 2)
        feat = self.conv(xs_t)
        if self.use_sigmoid:
            feat = torch.sigmoid(feat)
        out = torch.transpose(feat, 1, 2)  # B, T, C, H, W
        return out



class Discriminator2D(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator2D, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs

def spectral_norm(module, mode=True):
    if mode:
        return _spectral_norm(module)
    return module
