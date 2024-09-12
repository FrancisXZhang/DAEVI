import os
import glob
from tqdm import tqdm
import importlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter

from core.dataset import Dataset
from core.loss import AdversarialLoss, PerceptualLoss, StyleLoss
import sys

from depth_layers import disp_to_depth
from depth_model.depth_decoder import DepthDecoder
from depth_model.resnet_encoder import ResnetEncoder

import torchvision

epsilon=sys.float_info.epsilon


class Trainer():
    def __init__(self, config, debug=True):
        self.config = config
        self.epoch = 0
        self.iteration = 0
        if debug:
            self.config['trainer']['save_freq'] = 5
            self.config['trainer']['valid_freq'] = 5
            self.config['trainer']['iterations'] = 5

        # setup data set and data loader
        self.train_dataset = Dataset(config['data_loader'], split='train',  debug=debug)
        self.train_sampler = None
        self.train_args = config['trainer']
        if config['distributed']:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=config['world_size'], 
                rank=config['global_rank'])
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_args['batch_size'] // config['world_size'],
            shuffle=(self.train_sampler is None), 
            num_workers=self.train_args['num_workers'],
            sampler=self.train_sampler)

        # set loss functions 
        self.adversarial_loss = AdversarialLoss(type=self.config['losses']['GAN_LOSS'])
        self.adversarial_loss = self.adversarial_loss.to(self.config['device'])
        self.perceptual_loss = PerceptualLoss().to(self.config['device'])
        self.style_loss = StyleLoss().to(self.config['device'])
        self.l1_loss = nn.L1Loss()

        # setup models including generator and discriminator
        net = importlib.import_module('model.'+config['model'])
        self.netG = net.InpaintGenerator(in_channels=3)
        self.netG = self.netG.to(self.config['device'])
        self.netD = net.Discriminator(
            in_channels=4, use_sigmoid=config['losses']['GAN_LOSS'] != 'hinge')
        self.netD = self.netD.to(self.config['device'])
        self.optimG = torch.optim.Adam(
            self.netG.parameters(), 
            lr=config['trainer']['lr'],
            betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))
        self.optimD = torch.optim.Adam(
            self.netD.parameters(), 
            lr=config['trainer']['lr'],
            betas=(self.config['trainer']['beta1'], self.config['trainer']['beta2']))


        self.load_initialmodel() #added by Rema for loading initializing model
        self.load()

        if config['distributed']:
            self.netG = DDP(
                self.netG, 
                device_ids=[self.config['local_rank']], 
                output_device=self.config['local_rank'],
                broadcast_buffers=True, 
                find_unused_parameters=False)
            self.netD = DDP(
                self.netD, 
                device_ids=[self.config['local_rank']], 
                output_device=self.config['local_rank'],
                broadcast_buffers=True, 
                find_unused_parameters=False)

        # set summary writer
        self.dis_writer = None
        self.gen_writer = None
        self.summary = {}
        if self.config['global_rank'] == 0 or (not config['distributed']):
            self.dis_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'dis'))
            self.gen_writer = SummaryWriter(
                os.path.join(config['save_dir'], 'gen'))

    # get current learning rate
    def get_lr(self):
        return self.optimG.param_groups[0]['lr']

     # learning rate scheduler, step
    def adjust_learning_rate(self):
        decay = 0.1**(min(self.iteration,
                          self.config['trainer']['niter_steady']) // self.config['trainer']['niter'])
        new_lr = self.config['trainer']['lr'] * decay
        if new_lr != self.get_lr():
            for param_group in self.optimG.param_groups:
                param_group['lr'] = new_lr
            for param_group in self.optimD.param_groups:
                param_group['lr'] = new_lr

    # add summary
    def add_summary(self, writer, name, val):
        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        if writer is not None and self.iteration % 100 == 0:
            writer.add_scalar(name, self.summary[name]/100, self.iteration)
            self.summary[name] = 0

    # load netG and netD
    def load(self):
        model_path = self.config['save_dir']
        if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):
            latest_epoch = open(os.path.join(
                model_path, 'latest.ckpt'), 'r').read().splitlines()[-1]
        else:
            ckpts = [os.path.basename(i).split('.pth')[0] for i in glob.glob(
                os.path.join(model_path, '*.pth'))]
            ckpts.sort()
            latest_epoch = ckpts[-1] if len(ckpts) > 0 else None

        if latest_epoch is not None:
            gen_path = os.path.join(
                model_path, 'gen_{}.pth'.format(str(latest_epoch).zfill(5)))
            dis_path = os.path.join(
                model_path, 'dis_{}.pth'.format(str(latest_epoch).zfill(5)))
            opt_path = os.path.join(
                model_path, 'opt_{}.pth'.format(str(latest_epoch).zfill(5)))
            if self.config['global_rank'] == 0:
                print('Loading model from {}...'.format(gen_path))
            data = torch.load(gen_path, map_location=self.config['device'])
            self.netG.load_state_dict(data['netG'])
            data = torch.load(dis_path, map_location=self.config['device'])
            self.netD.load_state_dict(data['netD'])
            data = torch.load(opt_path, map_location=self.config['device'])
            self.optimG.load_state_dict(data['optimG'])
            self.optimD.load_state_dict(data['optimD'])
            self.epoch = data['epoch']
            self.iteration = data['iteration']

        else:
            if self.config['global_rank'] == 0:
                print(
                    'Warnning: There is no trained model found. An initialized model will be used.')

                
    def load_initialmodel(self):#added by Rema for loading initializing model
        if os.path.isfile(os.path.join(self.config['initialmodel'], 'latest.ckpt')):
            latest_epoch = open(os.path.join(
                self.config['initialmodel'], 'latest.ckpt'), 'r').read().splitlines()[-1]
            if latest_epoch is not None:
                gen_path = os.path.join(
                    self.config['initialmodel'], 'gen_{}.pth'.format((self.config['chosen_epoch']).zfill(5)))
                dis_path = os.path.join(
                    self.config['initialmodel'], 'dis_{}.pth'.format((self.config['chosen_epoch']).zfill(5)))
                opt_path = os.path.join(
                    self.config['initialmodel'], 'opt_{}.pth'.format((self.config['chosen_epoch']).zfill(5)))
                if self.config['global_rank'] == 0:
                    print('Loading model from {}...'.format(gen_path))
                data = torch.load(gen_path, map_location=self.config['device'])
                
                self.netG.load_state_dict(data['netG'])
                data = torch.load(dis_path, map_location=self.config['device'])
                self.netD.load_state_dict(data['netD'])
                data = torch.load(opt_path, map_location=self.config['device'])
                
                self.optimG.load_state_dict(data['optimG'])
                self.optimD.load_state_dict(data['optimD'])

                self.epoch = data['epoch']
                self.iteration = data['iteration']
            else:
                if self.config['global_rank'] == 0:
                    print(
                        'Warnning: There is no trained model found. An initialized model will be used.')
        if os.path.isfile(os.path.join(self.config['initialmodel'], 'sttn.pth')):
            latest_epoch=0
            gen_path = os.path.join(
                self.config['initialmodel'], 'sttn.pth')
            if self.config['global_rank'] == 0:
                print('Loading model from {}...'.format(gen_path))
            data = torch.load(gen_path, map_location=self.config['device'])
            self.netG.load_state_dict(data['netG'])
                
    # save parameters every eval_epoch
    def save(self, it):
        if self.config['global_rank'] == 0:
            gen_path = os.path.join(
                self.config['save_dir'], 'gen_{}.pth'.format(str(it).zfill(5)))
            dis_path = os.path.join(
                self.config['save_dir'], 'dis_{}.pth'.format(str(it).zfill(5)))
            opt_path = os.path.join(
                self.config['save_dir'], 'opt_{}.pth'.format(str(it).zfill(5)))
            print('\nsaving model to {} ...'.format(gen_path))
            if isinstance(self.netG, torch.nn.DataParallel) or isinstance(self.netG, DDP):
                netG = self.netG.module
                netD = self.netD.module
            else:
                netG = self.netG
                netD = self.netD
            torch.save({'netG': netG.state_dict()}, gen_path)
            torch.save({'netD': netD.state_dict()}, dis_path)
            torch.save({'epoch': self.epoch,
                        'iteration': self.iteration,
                        'optimG': self.optimG.state_dict(),
                        'optimD': self.optimD.state_dict()}, opt_path)
            os.system('echo {} > {}'.format(str(it).zfill(5),
                                            os.path.join(self.config['save_dir'], 'latest.ckpt')))

    # train entry
    def train(self):
        pbar = range(int(self.train_args['iterations']))
        if self.config['global_rank'] == 0:
            pbar = tqdm(pbar, initial=self.iteration, dynamic_ncols=True, smoothing=0.01)
        
        while True:
            self.epoch += 1
            if self.config['distributed']:
                self.train_sampler.set_epoch(self.epoch)

            self._train_epoch(pbar)
            if self.iteration > self.train_args['iterations']:
                break
        print('\nEnd training....')

    # process input and calculate loss every training epoch
    def _train_epoch(self, pbar):
        device = self.config['device']

        for frames, masks, depths in self.train_loader:
            self.adjust_learning_rate()
            self.iteration += 1

            frames, masks, depths = frames.to(device), masks.to(device), depths.to(device)
            b, t, c, h, w = frames.size()

            # rescale depth map to [-1, 1]
            scaled_dep = depths / max(torch.max(depths), epsilon) * 2 -1
            scaled_dep = scaled_dep.view(b*t, 1, h, w)

            # inpainting
            masked_frame = (frames * (1 - masks).float())
            pred_img, pred_dep = self.netG(masked_frame, masks)
            frames = frames.view(b*t, c, h, w)
            masks = masks.view(b*t, 1, h, w)
            
            comp_frame = torch.cat([frames, scaled_dep], dim=1)
            comp_frame = comp_frame.view(b, t, c+1, h, w)
            
            comp_img = torch.cat([pred_img, pred_dep], dim=1)
            comp_img = comp_img.view(b, t, c+1, h, w)
            
            # frames*(1.-masks) + masks*pred_img # changed to use the full predicted image

            gen_loss = 0
            dis_loss = 0

            # discriminator adversarial loss
            real_vid_feat = self.netD(comp_frame)
            fake_vid_feat = self.netD(comp_img.detach())
            dis_real_loss = self.adversarial_loss(real_vid_feat, True, True)
            dis_fake_loss = self.adversarial_loss(fake_vid_feat, False, True)
            dis_loss += (dis_real_loss + dis_fake_loss) / 2
            self.add_summary(
                self.dis_writer, 'loss/dis_vid_fake', dis_fake_loss.item())
            self.add_summary(
                self.dis_writer, 'loss/dis_vid_real', dis_real_loss.item())
            self.optimD.zero_grad()
            dis_loss.backward()
            self.optimD.step()

            # generator adversarial loss
            gen_vid_feat = self.netD(comp_img)
            gan_loss = self.adversarial_loss(gen_vid_feat, True, False)
            gan_loss = gan_loss * self.config['losses']['adversarial_weight']
            gen_loss += gan_loss
            self.add_summary(
                self.gen_writer, 'loss/gan_loss', gan_loss.item())

            # generator l1 loss
            hole_loss = self.l1_loss(pred_img*masks, frames*masks)
            hole_loss = hole_loss / max(torch.mean(masks), epsilon) * self.config['losses']['hole_weight']
            gen_loss += hole_loss 
            self.add_summary(
                self.gen_writer, 'loss/hole_loss', hole_loss.item())

            valid_loss = self.l1_loss(pred_img*(1-masks), frames*(1-masks))
            valid_loss = valid_loss / max(torch.mean(1-masks), epsilon) * self.config['losses']['valid_weight']
            gen_loss += valid_loss 
            self.add_summary(
                self.gen_writer, 'loss/valid_loss', valid_loss.item())

            # generator perceptual loss
            perceptual_loss = self.perceptual_loss(pred_img, frames)
            perceptual_loss = perceptual_loss * self.config['losses']['perceptual_weight']
            gen_loss += perceptual_loss

            # generator style loss
            style_loss = self.style_loss(pred_img*masks, frames*masks)
            style_loss = style_loss * self.config['losses']['style_weight']
            gen_loss += style_loss

            # depth loss
            depth_loss = self.l1_loss(pred_dep, scaled_dep)
            depth_loss = depth_loss / max(torch.mean(masks), epsilon) * self.config['losses']['depth_weight']
            gen_loss += depth_loss

            # optimize
            self.optimG.zero_grad()
            gen_loss.backward()
            self.optimG.step()

            # console logs
            if self.config['global_rank'] == 0:
                pbar.update(1)
                pbar.set_description((
                    f"d: {dis_loss.item():.3f}; g: {gan_loss.item():.3f};"
                    f"hole: {hole_loss.item():.3f}; valid: {valid_loss.item():.3f};"
                    f"perceptual: {perceptual_loss.item():.3f}"
                    f"style: {style_loss.item():.3f}"
                    f"depth: {depth_loss.item():.3f}"
                    )
                )


            # Saving the depth images for visualization
            if self.iteration % self.train_args['save_freq'] == 0:
                # Create directory for saving images if it doesn't exist
                save_dir = os.path.join(self.config['save_dir'], 'visualization')
                os.makedirs(save_dir, exist_ok=True)

                # Save images    
                masked_frame = masked_frame.view(b*t, c, h, w)
                comp_img = pred_img * masks + frames * (1 - masks)

                # map to 0-1
                pred_dep = pred_dep.view(b*t, 1, h, w)
                pred_dep = (pred_dep + 1) / 2 * max(torch.max(depths), epsilon)
                pred_dep, _ = disp_to_depth(pred_dep, 0.1, 150)

                depths = depths.view(b*t, 1, h, w)
                depths, _ = disp_to_depth(depths, 0.1, 150)

                img_grid = torchvision.utils.make_grid(torch.cat([pred_dep/255, depths/255], dim=0), nrow=b*t)
                torchvision.utils.save_image(img_grid, os.path.join(save_dir, f"{self.epoch}_{self.iteration}_depth.png"))
                img_grid = torchvision.utils.make_grid(torch.cat([frames, masked_frame, pred_img, comp_img], dim=0), nrow=b*t)
                torchvision.utils.save_image(img_grid, os.path.join(save_dir, f"{self.epoch}_{self.iteration}_inpainting.png"))


            # saving models
            if self.iteration % self.train_args['save_freq'] == 0:
                self.save(int(self.iteration//self.train_args['save_freq']))
            if self.iteration > self.train_args['iterations']:
                break

