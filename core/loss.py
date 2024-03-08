import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from depth_model.resnet_encoder import ResnetEncoder


class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()
        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()
        elif type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):

        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()
        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(
                outputs)
            loss = self.criterion(outputs, labels)
            return loss



class StyleLoss(nn.Module):

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss



class PerceptualLoss(nn.Module):

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])


        return content_loss



class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }

        return out



class Depth_Aware_Loss(torch.nn.Module):


    def __init__(self, config, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(Depth_Aware_Loss, self).__init__()
        
        self.config = config

        self.criterion = torch.nn.L1Loss()
        self.weights = weights


        self.depth_encoder = ResnetEncoder(18, False, 1)
        
        encoder_path = os.path.join(self.config['trainer']['depth_model_path'], "encoder.pth")
        loaded_dict_enc = torch.load(encoder_path, map_location=self.config['device'])

        # extract the height and width of image that this model was trained with
        self.depth_feed_height = loaded_dict_enc['height']
        self.depth_feed_width = loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.depth_encoder.state_dict()}

        self.depth_encoder.load_state_dict(filtered_dict_enc)
        self.depth_encoder.to(self.config['device'])
        self.depth_encoder.eval()

    def __call__(self, x, y):
        
        # resize input images to feed into depth encoder
        x = nn.functional.interpolate(x, size=(self.depth_feed_height, self.depth_feed_width), mode='bilinear', align_corners=False)
        y = nn.functional.interpolate(y, size=(self.depth_feed_height, self.depth_feed_width), mode='bilinear', align_corners=False)

        x_depth = self.depth_encoder(x)
        y_depth = self.depth_encoder(y)

        depth_loss = 0.0

        depth_loss += self.weights[0] * self.criterion(x_depth[0], y_depth[0])
        depth_loss += self.weights[1] * self.criterion(x_depth[1], y_depth[1])
        depth_loss += self.weights[2] * self.criterion(x_depth[2], y_depth[2])
        depth_loss += self.weights[3] * self.criterion(x_depth[3], y_depth[3])
        depth_loss += self.weights[4] * self.criterion(x_depth[4], y_depth[4])
        
        return depth_loss


class TransformationLoss(nn.Module):
    def __init__(self):
        super(TransformationLoss, self).__init__()

    def forward(self, warped_image, target_image):
        # Compute L1 or any other suitable loss
        loss = F.l1_loss(warped_image, target_image)
        return loss


def compute_transformation_loss_for_batch(images, depths, poses, transformation_loss_module):
    total_loss = 0.0
    for i in range(images.size(0) - 1):  # Loop through each pair of consecutive frames
        # Compute the relative pose between consecutive frames
        relative_pose = compute_relative_pose(poses[i], poses[i + 1])

        # Warp the current image to the next time step using the estimated depth and relative pose
        warped_image = warp_image(images[i], depths[i], relative_pose)

        # Compute the transformation loss between the warped image and the actual next frame
        loss = transformation_loss_module(warped_image, images[i + 1])

        # Accumulate the loss
        total_loss += loss

    # Return the average loss over the batch
    return total_loss / (images.size(0) - 1)

