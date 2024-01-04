import os
import time
import datetime
import logging
import argparse
import numpy as np
import skvideo.io
import cv2
import trimesh
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.datasets.style_field_dataset import StyleFieldDataset
from torch.utils.data import DataLoader
from munch import *
from pytorch3d.structures import Meshes
# from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
# from icecream import ic
from tqdm import tqdm
import glob
from PIL import Image
import torchvision.transforms as transforms

from termcolor import colored
import mcubes
import trimesh
import time
import pytorch_lightning as pl
from options import BaseOptions
from model import Generator, Discriminator
from model import Generator
from torchvision.utils import make_grid, save_image
import torch.nn.functional as F
from lib.losses.perceptual_loss import PerceptualLoss
from lib.losses import vgg_feat
from lib.losses.elastic_loss import JacobianSmoothness
from torch.utils.tensorboard import SummaryWriter
from lib.utils.recorder import make_recorder
from termcolor import colored
from utils import (
    generate_camera_params, align_volume, extract_mesh_with_marching_cubes,
    xyz2mesh, create_cameras, create_mesh_renderer, add_textures, requires_grad
    )
from losses import d_logistic_loss, d_r1_loss, g_nonsaturating_loss
from pdb import set_trace as st

class ModelWrapper(nn.Module):
    def __init__(self, opt, generator, device='cuda'):
        super().__init__()
        self.opt = opt
        self.mean_latent = None
        self.generator = generator
        self.device = device
        mean_latent = self.generator.mean_latent(opt.inference.truncation_mean, device=self.device)
        self.mean_latent = [itm.clone().detach().cpu() for itm in mean_latent]

    def forward(self, batch):
        if isinstance(batch['z'], list):
            noise = batch['z'] # [torch.Size([1, 9, 256]), torch.Size([1, 10, 512])] W+
            is_chunk = False
        else:
            noise = batch['z'].unsqueeze(1) # [torch.Size([1, 256])
            is_chunk = True
        cam_extrinsics = batch['cam_extrinsics']
        focal = batch['focal']
        near = batch['near']
        far = batch['far']
        # location = batch['location']
        # real_img = batch['real_img']
        # cartoon_img = batch['cartoon_img']

        gen_imgs_ls = []
        grad_style_ls = []
        dx_norm_ls = []
        

        for j in range(0, cam_extrinsics.shape[0], self.opt.style.style_chunk):
            
            ### assume chunk size is 1
            if is_chunk:
                # curr_noise = [n[j:j+self.opt.style.style_chunk] for n in noise]
                exstyle = batch['exstyle'][j:j+self.opt.style.style_chunk]
                if 'exstyle_g0' in batch:
                    exstyle_g0 = batch['exstyle_g0'][j:j+self.opt.style.style_chunk]
                else:
                    exstyle_g0 = None
                if 'exstyle_g1' in batch:
                    exstyle_g1 = batch['exstyle_g1'][j:j+self.opt.style.style_chunk]
                else:
                    exstyle_g1 = None
                gen_imgs, _, others = self.generator(noise[j:j+self.opt.style.style_chunk],
                                    cam_extrinsics[j:j+self.opt.style.style_chunk],
                                    focal[j:j+self.opt.style.style_chunk],
                                    near[j:j+self.opt.style.style_chunk],
                                    far[j:j+self.opt.style.style_chunk],
                                    truncation=self.opt.inference.truncation_ratio,
                                    truncation_latent=[itm.to(self.device) for itm in self.mean_latent],
                                    inversion_latent_type=self.opt.inference.inversion_latent_type,
                                    input_is_latent=self.opt.inference.input_is_latent,
                                    exstyle=exstyle, exstyle_g0=exstyle_g0, exstyle_g1=exstyle_g1)
            else:
                exstyle = batch['exstyle']
                if 'exstyle_g0' in batch:
                    exstyle_g0 = batch['exstyle_g0']
                else:
                    exstyle_g0 = None
                if 'exstyle_g1' in batch:
                    exstyle_g1 = batch['exstyle_g1']
                else:
                    exstyle_g1 = None
                gen_imgs, _, others = self.generator(noise,
                                    cam_extrinsics,
                                    focal,
                                    near,
                                    far,
                                    truncation=self.opt.inference.truncation_ratio,
                                    truncation_latent=[itm.to(self.device) for itm in self.mean_latent],
                                    inversion_latent_type=self.opt.inference.inversion_latent_type,
                                    input_is_latent=self.opt.inference.input_is_latent,
                                    exstyle=exstyle, exstyle_g0=exstyle_g0, exstyle_g1=exstyle_g1)
            
            
            gen_imgs_ls.append(gen_imgs)
            grad_style_ls.append(others['grad_style'])
            dx_norm_ls.append(others['dx_norm'])
        
        gen_img_batch = torch.cat(gen_imgs_ls, dim=0)
        if self.opt.style.elastic_loss > 0:
            grad_style_batch = torch.cat(grad_style_ls, dim=0)
        else:
            grad_style_batch = None
        dx_norm_batch = torch.cat(dx_norm_ls, dim=0)
        # shift value range from [-1,1] to [0,1] for gen image
        gen_img_batch = (gen_img_batch + 1) / 2
        
        ret = {'gen_img_batch': gen_img_batch, 'grad_style_batch': grad_style_batch, 'dx_norm_batch': dx_norm_batch}
        ret.update(others)
        return ret
    
class Runner(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.device = 'cuda'
        self.dataset = StyleFieldDataset(data_dir=self.opt.style.data_dir, source_data=self.opt.style.source_data, style_data=self.opt.style.style_data, train_split_ratio=self.opt.style.train_split_ratio)
        self.train_loader = DataLoader(self.dataset, batch_size=self.opt.style.style_batch, num_workers=self.opt.style.num_workers, shuffle=True, pin_memory=True)
        print('train loader size:', len(self.train_loader.dataset))
        self.opt.style.n_styles = max(self.dataset.n_styles, self.opt.style.n_styles)
        print('n_styles:', self.opt.style.n_styles)

        self.construct_modules()
        if self.opt.experiment.exp_mode == 'train':
            self.define_losses()
            self.configure_optimizers()
            self.model = nn.DataParallel(ModelWrapper(self.opt, self.generator, device=self.device).to(self.device))

            if self.opt.style.gan_loss > 0:
                self.discriminator = nn.DataParallel(self.discriminator.to(self.device))
            ### tensorboard
            self.checkpoint_path = os.path.join(self.opt.experiment.root_save_dir, 'training_records', self.opt.style.jobname )
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
            os.makedirs(self.checkpoint_path, exist_ok=True)
            writer = SummaryWriter(log_dir=self.checkpoint_path)
            self.recorder = make_recorder(writer)
        elif 'visual_surface' == self.opt.experiment.exp_mode:
            print('visual surface mode')
        else:
            self.model = nn.DataParallel(ModelWrapper(self.opt, self.generator, device=self.device).to(self.device))
    
    def construct_modules(self):
        self.exstyle_mapper = torch.nn.Embedding(self.opt.style.n_styles, self.opt.style.n_dim_exstyle).to(self.device)
         # define model
        self.generator = Generator(self.opt.model, self.opt.rendering, style_opt=self.opt.style).to(self.device)
        
        # load parameters NOTE: able to load part of the model stored in pre-trained SDF
        
        checkpoint_path = os.path.join(self.opt.experiment.root_save_dir, 'full_models', self.opt.experiment.expname + '.pt')
        checkpoint = torch.load(checkpoint_path)
        pretrained_weights_dict = checkpoint["g_ema"]
        model_dict = self.generator.state_dict()
        
        count_load = 0
        count_total = 0
        for k, v in pretrained_weights_dict.items():
            count_total +=1
            if v.size() == model_dict[k].size():
                model_dict[k] = v
                count_load += 1
        self.generator.load_state_dict(model_dict)
        print(colored('Loaded pretrained StyleSDF, total params: {}, loaded params: {}'.format(count_total, count_load), 'red'))
        mean_latent = self.generator.mean_latent(self.opt.inference.truncation_mean, device=self.device)
        self.mean_latent = [itm.clone().detach() for itm in mean_latent]

        if self.opt.style.gan_loss > 0:
            self.discriminator = Discriminator(self.opt.model).to(self.device)
            self.discriminator.load_state_dict(checkpoint["d"])
        


        self.begin_epoch = 0
        
        if self.opt.experiment.exp_mode != 'train':
            self.load_checkpoint()
            return

        # define parameters to train or fine-tune
        # default all parameters do not require grad
        for param in self.generator.parameters():
            param.requires_grad = False
        if self.opt.style.style_field:
            # self.train_nets =[self.generator.renderer.style_field]
            self.train_nets ={
                'style_field': self.generator.renderer.style_field,
            }
            for param in self.generator.renderer.style_field.parameters():
                param.requires_grad = True
            if self.opt.style.adaptive_style_mixing and len(self.opt.style.adaptive_style_mixing_blocks) > 0:
                self.train_nets['decoder_res'] = self.generator.decoder.res
                self.train_nets['decoder_t_c'] = self.generator.decoder.t_c
                for param in self.generator.decoder.res.parameters():
                    param.requires_grad = True
                for param in self.generator.decoder.t_c.parameters():
                    param.requires_grad = True
        else:
            raise
        
        if len(self.opt.style.finetune_nets) > 0:
            assert not self.opt.style.adaptive_style_mixing, "Do not fine-tune decoder in this mode!!!"
            self.finetune_nets = {}
            nets_to_require_grad = []
            finetune_net_names = self.opt.style.finetune_nets.split('+')
            # specified blocks in decoder to fine-tune
            if 'conv' in self.opt.style.finetune_nets:
                self.finetune_nets['decoder'] = self.generator.decoder
                if 'conv4' in finetune_net_names:
                    nets_to_require_grad += [self.generator.decoder.convs[6:8] + self.generator.decoder.to_rgbs[3:4]]
                if 'conv3' in finetune_net_names:
                    nets_to_require_grad += [self.generator.decoder.convs[4:6] + self.generator.decoder.to_rgbs[2:3]]
                if 'conv2' in finetune_net_names:
                    nets_to_require_grad += [self.generator.decoder.convs[2:4] + self.generator.decoder.to_rgbs[1:2]]
                if 'conv1' in finetune_net_names:
                    nets_to_require_grad += [self.generator.decoder.convs[0:2] + self.generator.decoder.to_rgbs[0:1]]
                if 'conv0' in finetune_net_names:
                    nets_to_require_grad += [self.generator.decoder.conv1 + self.generator.decoder.to_rgb1]
            # specified blocks in renderer.network to fine-tune
            if 'linear' in self.opt.style.finetune_nets:
                self.finetune_nets['renderer_network'] = self.generator.renderer.network
                nets_to_require_grad += [getattr(self.generator.renderer.network,itm) for itm in finetune_net_names if 'linear' in itm]
            for sub_net in nets_to_require_grad:
                for param in sub_net.parameters():
                    param.requires_grad = True
    
    def define_losses(self):
        if self.opt.style.l1_loss > 0: 
            self.l1_loss_fn = nn.L1Loss()
        if self.opt.style.percep_loss > 0:
            self.percep_loss_fn = PerceptualLoss()
        if self.opt.style.elastic_loss > 0:
            self.elastic_loss_fn = JacobianSmoothness().to(self.device)

    def configure_optimizers(self):
        params_to_train = []
        for key, val in self.train_nets.items():
            params_to_train += list(val.parameters())
        # finetune
        if len(self.opt.style.finetune_nets) > 0:
            params_to_finetune = []
            for key, val in self.finetune_nets.items():
                params_to_finetune += list(val.parameters())
            params_to_finetune_ls = [{'params': itm, 'lr': self.opt.style.finetune_lr} for itm in params_to_finetune]
            params_to_train += params_to_finetune

        self.optimizer = torch.optim.Adam(params_to_train, 
            lr=self.opt.style.train_lr, betas=(0.9, 0.999), weight_decay=self.opt.style.weight_decay)
       
        if self.opt.style.gan_loss > 0:
            self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), 
                lr=self.opt.style.d_lr, betas=(0.9, 0.999), weight_decay=self.opt.style.weight_decay)
            
    def compute_losses(self, cartoon_img, forward_output):
        self.loss_stats = {}
        loss = 0
        gen_img_batch = forward_output['gen_img_batch']
        if self.opt.style.l1_loss > 0: 
            l1_loss = self.l1_loss_fn(gen_img_batch, cartoon_img)
            self.loss_stats.update({"train/l1_loss": l1_loss})
            loss += self.opt.style.l1_loss * l1_loss
        if self.opt.style.percep_loss > 0:
            percep_loss = self.percep_loss_fn(gen_img_batch, cartoon_img)
            percep_loss = percep_loss.mean()
            self.loss_stats.update({"train/percept_loss": percep_loss})
            loss += self.opt.style.percep_loss * percep_loss
        if self.opt.style.elastic_loss > 0:
            elastic_loss = self.elastic_loss_fn(forward_output['grad_style_batch'])
            self.loss_stats.update({"train/elastic_loss": elastic_loss})
            loss += self.opt.style.elastic_loss * elastic_loss
        # log dx_norm for analysis only 
        if 'dx_norm_batch' in forward_output:
            self.loss_stats.update({"train/dx_norm": forward_output['dx_norm_batch']})
        if self.opt.style.gan_loss > 0:
            g_gan_loss = g_nonsaturating_loss(forward_output['fake_pred'])
            loss += self.opt.style.gan_loss * g_gan_loss
            self.loss_stats.update({"train/g_gan_loss": g_gan_loss})
        return loss
    
    def to_cuda(self, batch):
        if isinstance(batch, tuple) or isinstance(batch, list):
            batch = [self.to_cuda(b) for b in batch]
            return batch
        # dict
        for k in batch:
            if k == 'meta':
                continue
            if isinstance(batch[k], tuple) or isinstance(batch[k], list):
                pass
            else:
                batch[k] = batch[k].to(self.device)

        return batch

    def train(self):
        print('start training from epoch', self.begin_epoch)
        self.max_iter = len(self.train_loader)
        self.recorder.step = self.begin_epoch * self.max_iter
        end = time.time()
        for epoch in range(self.begin_epoch, self.opt.style.n_epoch):
            self.recorder.epoch = epoch
            for iteration, batch in tqdm(enumerate(self.train_loader)):

                data_time = time.time() - end
                batch = self.to_cuda(batch)
                if self.opt.style.gan_loss == 0:
                    batch['exstyle'] = self.exstyle_mapper(batch['style_id']) # (B, 256)
                    forward_output = self.model(batch)
                    loss = self.compute_losses(batch['cartoon_img'], forward_output)
                    self.optimizer.zero_grad()
                    loss = loss.mean()
                    loss.backward()
                    self.optimizer.step()
                    end = self.train_utils(epoch, iteration, end, data_time, batch, forward_output)
                else:
                    data_time = time.time() - end
                    batch = self.to_cuda(batch)
                    if iteration % 2 == 0:
                        # D loop
                        requires_grad(self.discriminator, True)
                        self.set_generator_grad(False)
                        self.discriminator.zero_grad()
                        d_regularize = self.recorder.step % self.opt.training.d_reg_every == 0
                        batch['exstyle'] = self.exstyle_mapper(batch['style_id']) # (B, 256)
                        fake_output = self.model(batch)
                        fake_pred = self.discriminator(fake_output['gen_img_batch'].detach())
                        if d_regularize:
                            batch['cartoon_img'].requires_grad = True
                        real_pred = self.discriminator(batch['cartoon_img'])
                        d_gan_loss = d_logistic_loss(real_pred, fake_pred)
                        if d_regularize:
                            grad_penalty = d_r1_loss(real_pred, batch['cartoon_img'])
                            r1_loss = self.opt.training.r1 * 0.5 * grad_penalty * self.opt.training.d_reg_every
                        else:
                            r1_loss = torch.tensor(0.0, device=self.device)
                        d_loss = d_gan_loss + r1_loss
                        self.loss_stats = {
                            'D/d_gan_loss': d_gan_loss.mean(),
                            'D/d_r1_loss': r1_loss.mean(),
                            'D/real_score': real_pred.mean(),
                            'D/fake_score': fake_pred.mean(),
                        }
                        d_loss.backward()
                        self.optimizer_D.step()
                        end = self.train_utils(epoch, iteration, end, data_time, batch, fake_output)
                    else:
                        # G loop
                        requires_grad(self.discriminator, False)
                        self.set_generator_grad(True)
                        self.generator.zero_grad()
                        batch['exstyle'] = self.exstyle_mapper(batch['style_id']) # (B, 256)
                        fake_output = self.model(batch)
                        fake_pred = self.discriminator(fake_output['gen_img_batch'])
                        fake_output['fake_pred'] = fake_pred
                        loss = self.compute_losses(batch['cartoon_img'], fake_output)
                        loss = loss.mean()
                        loss.backward()
                        self.optimizer.step()
                        end = self.train_utils(epoch, iteration, end, data_time, batch, fake_output)
                              
            if (epoch + 1) % self.opt.style.save_checkpoint_interval == 0 :
                self.save_checkpoint(epoch)
            if (epoch + 1) % self.opt.style.save_latest_checkpoint_interval == 0 :
                self.save_checkpoint(epoch, last=True)
                
    def train_utils(self, epoch, iteration, end, data_time, batch, forward_output):
        # log related
        self.loss_stats = self.reduce_loss_stats(self.loss_stats)
        self.recorder.update_loss_stats(self.loss_stats)
        batch_time = time.time() - end
        end = time.time()
        self.recorder.batch_time.update(batch_time)
        self.recorder.data_time.update(data_time)
        self.recorder.step +=1
        self.recorder.record('train')
        # print training state
        if iteration % (self.opt.style.log_interval+1) == 0 or iteration == (self.max_iter - 1):
            # print training state
            eta_seconds = self.recorder.batch_time.global_avg * (self.max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            # lr = self.optimizer['lr']
            memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
            training_state = '  '.join(['eta: {}', '{}', 'max_mem: {:.0f}'])
            training_state = training_state.format(eta_string, str(self.recorder), memory)
            print(training_state)
        # visualize
        if (self.opt.style.style_batch * self.recorder.step - self.begin_epoch * self.max_iter * self.opt.style.style_batch) < 4096:
            vis_flag = (self.opt.style.style_batch * self.recorder.step)  % self.opt.style.vis_interval == 0
        else:
            vis_flag = (self.opt.style.style_batch * self.recorder.step)  % 4096 == 0
        if vis_flag:
            real_img, cartoon_img, gen_img_batch = batch['real_img'], batch['cartoon_img'], forward_output['gen_img_batch']
            # visualize
            vis_size = self.opt.style.vis_size
            real_img = F.interpolate(real_img, size=(vis_size, vis_size), mode="nearest")
            cartoon_img = F.interpolate(cartoon_img, size=(vis_size, vis_size), mode="nearest")
            gen_img_batch_resize = F.interpolate(gen_img_batch, size=(vis_size, vis_size), mode="nearest").to(real_img.device)
            images = torch.cat([real_img, cartoon_img, gen_img_batch_resize], 0)
            grid = make_grid(images, nrow=real_img.shape[0])
            save_img_dir = os.path.join(self.opt.experiment.root_save_dir, 'vis', self.opt.style.jobname)
            if not os.path.exists(save_img_dir):
                os.makedirs(save_img_dir)
            save_image(grid, '{}/{}.png'.format(save_img_dir, self.opt.style.style_batch * self.recorder.step))
            print('img_saved for epoch {} batch {}'.format(epoch, iteration))
        return end

    def set_generator_grad(self, flag):
        "set on/off for sub modules of generator that need to train"
        for k, sub_module in self.train_nets.items():
            for param in sub_module.parameters():
                param.requires_grad = flag
    
    def load_checkpoint(self):
        if self.opt.experiment.continue_training > -1 :
            checkpoint_path = os.path.join(self.opt.experiment.root_save_dir, 'training_records', self.opt.style.jobname, 'checkpoints', f'{self.opt.experiment.continue_training}.pth')
        else:
            checkpoint_path = os.path.join(self.opt.experiment.root_save_dir, 'training_records', self.opt.style.jobname, 'checkpoints', 'latest.pth')
        checkpoint = torch.load(checkpoint_path)
        pretrained_model = checkpoint['generator']
        self.begin_epoch = checkpoint['epoch']
        print(colored(f'Loaded checkpoint from:, {checkpoint_path}', 'red'))

        model_dict = self.generator.state_dict()
        for k, v in pretrained_model.items():
            if k.startswith('generator'):
                k = k.replace('generator.', '')        
            model_dict[k] = v
        self.generator.load_state_dict(model_dict)
        
        if 'surface' in self.opt.experiment.exp_mode:
            self.opt['surf_extraction'] = Munch()
            self.opt.model.is_test = True
            self.opt.model.style_dim = 256
            self.opt.model.freeze_renderer = False
            self.opt.inference.size = self.opt.model.size
            self.opt.inference.camera = self.opt.camera
            self.opt.inference.renderer_output_size = self.opt.model.renderer_spatial_output_dim
            self.opt.inference.style_dim = self.opt.model.style_dim
            self.opt.inference.project_noise = self.opt.model.project_noise
            self.opt.rendering.perturb = 0
            self.opt.rendering.force_background = True
            self.opt.rendering.static_viewdirs = True
            self.opt.rendering.return_sdf = True
            self.opt.rendering.N_samples = 64
            self.opt.surf_extraction.rendering = self.opt.rendering
            self.opt.surf_extraction.model = self.opt.model.copy()
            self.opt.surf_extraction.model.renderer_spatial_output_dim = 128
            self.opt.surf_extraction.rendering.N_samples = self.opt.surf_extraction.model.renderer_spatial_output_dim
            self.opt.surf_extraction.rendering.return_xyz = True
            self.opt.surf_extraction.rendering.return_sdf = True
            self.opt.inference.surf_extraction_output_size = self.opt.surf_extraction.model.renderer_spatial_output_dim
            self.surface_g = Generator(self.opt.surf_extraction.model, self.opt.surf_extraction.rendering, opt.style, full_pipeline=False).to(self.device)
            model_dict2 = self.surface_g.state_dict()
            for k, v in pretrained_model.items():
                if k.startswith('generator'):
                    k = k.replace('generator.', '') 
                if k in model_dict2:
                    model_dict2[k] = v
            self.surface_g.load_state_dict(model_dict2)

        return     
    
    def save_checkpoint(self, epoch, last=False):        
        os.makedirs(os.path.join(self.checkpoint_path, 'checkpoints'), exist_ok=True)
        model = {
            # 'net': net.state_dict(),
            'optim': self.optimizer.state_dict(),
            # 'scheduler': scheduler.state_dict(),
            'recorder': self.recorder.state_dict(),
            'epoch': epoch,
            'generator': self.generator.state_dict(),
        }
        if self.opt.style.gan_loss > 0:
            model['discriminator'] = self.discriminator.state_dict()

        if last:
            torch.save(model, os.path.join(self.checkpoint_path, 'checkpoints', 'latest.pth'))
            print(f'saved latest.pth')
        else:
            torch.save(model, os.path.join(self.checkpoint_path, 'checkpoints', '{}.pth'.format(epoch)))
            print(f'saved {epoch}.pth')
    
    def evaluate(self):
        raise NotImplementedError
    
    def reduce_loss_stats(self, loss_stats):
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        return reduced_losses
    
    def prepare_subj_ls(self, return_latent_dict=True):
        subj_ls = []
        ranges = opt.inference.given_subject_list.split(',')
        for this_range in ranges:
            if '-' in this_range:
                min_subj, max_subj = this_range.split('-')
                for i in range(int(min_subj), int(max_subj)):
                    subj_ls.append(int(i))
                # subj_ls.extend(range.split('-'))
            else:
                subj_ls.append(int(this_range))
        print(f'subject list len: {len(subj_ls)}, {subj_ls[0]}-{subj_ls[-1]}')
        if not return_latent_dict:
            return subj_ls
        tic = time.time()
        latent_filepath = os.path.join(self.opt.style.data_dir, f'{self.opt.style.latents_eval}')
        latent_dict = torch.load(latent_filepath)
        toc = time.time()
        print('latent_dict done in {} sec'.format(toc-tic))
        return subj_ls, latent_dict

    def prepare_batch_from_latent(self, subj, latent_dict):
        subject_id = str(subj).zfill(7)
        # print('retriving subject id: ', subject_id, ' from latent_dict')
        subject_latent = latent_dict[subject_id]
        batch = {
            'z': subject_latent['z'][None],
            'cam_extrinsics': subject_latent['cam_extrinsics'][None],
                'focal':subject_latent['focal'][None],
                'near': subject_latent['near'][None],
                'far': subject_latent['far'][None],
        } 
        batch = self.to_cuda(batch)
        return batch

    def visualize_video(self):
        surface_mean_latent = [self.mean_latent[0].clone().detach()]
        num_frames = self.opt.inference.num_frames
        # Generate video trajectory
        trajectory = np.zeros((num_frames,3), dtype=np.float32)

        # set camera trajectory
        # sweep azimuth angles (4 seconds)
        if self.opt.inference.azim_video:
            t = np.linspace(0, 1, num_frames)
            elev = 0
            fov = opt.camera.fov
            if opt.camera.uniform:
                azim = self.opt.camera.azim * np.cos(t * 2 * np.pi)
            else:
                azim = 1.5 * self.opt.camera.azim * np.cos(t * 2 * np.pi)

            trajectory[:num_frames,0] = azim
            trajectory[:num_frames,1] = elev
            trajectory[:num_frames,2] = fov

        # elipsoid sweep (4 seconds)
        else:
            t = np.linspace(0, 1, num_frames)
            fov = self.opt.camera.fov #+ 1 * np.sin(t * 2 * np.pi)
            if self.opt.camera.uniform:
                elev = self.opt.camera.elev / 2 + self.opt.camera.elev / 2  * np.sin(t * 2 * np.pi)
                azim = self.opt.camera.azim  * np.cos(t * 2 * np.pi)
            else:
                elev = 1.5 * self.opt.camera.elev * np.sin(t * 2 * np.pi)
                azim = 1.5 * self.opt.camera.azim * np.cos(t * 2 * np.pi)

            trajectory[:num_frames,0] = azim
            trajectory[:num_frames,1] = elev
            trajectory[:num_frames,2] = fov

        trajectory = torch.from_numpy(trajectory).to(self.device)

        # generate input parameters for the camera trajectory
        # sample_cam_poses, sample_focals, sample_near, sample_far = \
        # generate_camera_params(trajectory, opt.renderer_output_size, device, dist_radius=opt.camera.dist_radius)
        sample_cam_extrinsics, sample_focals, sample_near, sample_far, _ = \
        generate_camera_params(self.opt.training.renderer_output_size, self.device, locations=trajectory[:,:2],
                            fov_ang=trajectory[:,2:], dist_radius=self.opt.camera.dist_radius)
        # # create geometry renderer (renders the depth maps)
        # cameras = create_cameras(azim=np.rad2deg(trajectory[0,0].cpu().numpy()),
        #                         elev=np.rad2deg(trajectory[0,1].cpu().numpy()),
        #                         dist=1, device=self.device)
        # renderer = create_mesh_renderer(cameras, image_size=512, specular_color=((0,0,0),),
        #                 ambient_color=((0.1,.1,.1),), diffuse_color=((0.75,.75,.75),),
        #                 device=self.device)
        
        parent_dir = os.path.join(self.opt.experiment.root_save_dir, 'visual/_ours')
        method_dir = os.path.join(parent_dir, f'video_{self.opt.style.jobname}')
        os.makedirs(parent_dir, exist_ok=True)
        os.makedirs(method_dir, exist_ok=True)
        style_id = self.opt.style.style_id
        subj_ls, latent_dict = self.prepare_subj_ls() 
        for subj in subj_ls:
            # print('working on subj', subj, type(subj))
            batch = self.prepare_batch_from_latent(subj, latent_dict)

            # input view 
            self.opt.camera.given_azim, self.opt.camera.given_elev = 0, 0
            cam = self.given_2camsettings()
            input_cam_extrinsics, input_focals, input_near, input_far, _ = cam
            batch['style_id'] = torch.Tensor([0]).long().cuda()
            batch['exstyle'] = self.exstyle_mapper(batch['style_id']) # (B, 256)
            batch['cam_extrinsics'] = input_cam_extrinsics
            batch['focal'] = input_focals
            batch['near'] = input_near
            batch['far'] = input_far
            with torch.no_grad():
                forward_output = self.model(batch)
                save_ls = [forward_output['gen_img_batch']]
                stem = '{}_source'.format(subj)
                self.save_image(batch, save_ls, stem=stem, save_dir=method_dir)
                del forward_output

            # video
            video_filename = '{}_s{}_image.mp4'.format(subj, style_id)
            save_ls = []
            writer = skvideo.io.FFmpegWriter(os.path.join(method_dir, video_filename),
                                         outputdict={'-pix_fmt': 'yuv420p', '-crf': '10'})
            for view in tqdm(range(num_frames)):
                batch['style_id'] = torch.Tensor([style_id]).long().cuda()
                batch['exstyle'] = self.exstyle_mapper(batch['style_id']) # (B, 256)
                batch['cam_extrinsics'] = sample_cam_extrinsics[view:view+1]
                batch['focal'] = sample_focals[view:view+1]
                batch['near'] = sample_near[view:view+1]
                batch['far'] = sample_far[view:view+1]
                with torch.no_grad():
                    forward_output = self.model(batch)
                    save_ls.append(forward_output['gen_img_batch'])
                    del forward_output
           
            images = torch.cat(save_ls, dim=0)
            rgb = 255 * (images.clamp(-1,1).permute(0,2,3,1).cpu().numpy())
            for k in range(num_frames):
                writer.writeFrame(rgb[k])
            writer.close()

    def visualize_surface(self):
        surface_mean_latent = [self.mean_latent[0].clone().detach()]
        num_frames = self.opt.inference.num_frames
        # Generate video trajectory
        trajectory = np.zeros((num_frames,3), dtype=np.float32)

        # set camera trajectory
        # sweep azimuth angles (4 seconds)
        if self.opt.inference.azim_video:
            t = np.linspace(0, 1, num_frames)
            elev = 0
            fov = opt.camera.fov
            if opt.camera.uniform:
                azim = self.opt.camera.azim * np.cos(t * 2 * np.pi)
            else:
                azim = 1.5 * self.opt.camera.azim * np.cos(t * 2 * np.pi)

            trajectory[:num_frames,0] = azim
            trajectory[:num_frames,1] = elev
            trajectory[:num_frames,2] = fov

        # elipsoid sweep (4 seconds)
        else:
            t = np.linspace(0, 1, num_frames)
            fov = self.opt.camera.fov #+ 1 * np.sin(t * 2 * np.pi)
            if self.opt.camera.uniform:
                elev = self.opt.camera.elev / 2 + self.opt.camera.elev / 2  * np.sin(t * 2 * np.pi)
                azim = self.opt.camera.azim  * np.cos(t * 2 * np.pi)
            else:
                elev = 1.5 * self.opt.camera.elev * np.sin(t * 2 * np.pi)
                azim = 1.5 * self.opt.camera.azim * np.cos(t * 2 * np.pi)

            trajectory[:num_frames,0] = azim
            trajectory[:num_frames,1] = elev
            trajectory[:num_frames,2] = fov

        trajectory = torch.from_numpy(trajectory).to(self.device)

        sample_cam_extrinsics, sample_focals, sample_near, sample_far, _ = \
        generate_camera_params(self.opt.training.renderer_output_size, self.device, locations=trajectory[:,:2],
                            fov_ang=trajectory[:,2:], dist_radius=self.opt.camera.dist_radius)
        
        parent_dir = os.path.join(self.opt.experiment.root_save_dir, 'visual/_ours')
        method_dir = os.path.join(parent_dir, f'video_{self.opt.style.jobname}')
        os.makedirs(parent_dir, exist_ok=True)
        os.makedirs(method_dir, exist_ok=True)
        style_id = self.opt.style.style_id
        subj_ls, latent_dict = self.prepare_subj_ls() 
        for subj in subj_ls:
            batch = self.prepare_batch_from_latent(subj, latent_dict)
            # depth video
            depth_filename = '{}_s{}_surface.mp4'.format(subj, style_id)
            mesh_ls = []
            depth_writer = skvideo.io.FFmpegWriter(os.path.join(method_dir, depth_filename),
                                outputdict={'-pix_fmt': 'yuv420p', '-crf': '1'})

            scale = 2.0
            sample_z = batch['z'][0:1]
            for view in tqdm(range(num_frames)):
                batch['style_id'] = torch.Tensor([style_id]).long().cuda()
                batch['exstyle'] = self.exstyle_mapper(batch['style_id']) # (B, 256)
                surface_out = self.surface_g([sample_z],
                                                sample_cam_extrinsics[view:view+1],
                                                sample_focals[view:view+1],
                                                sample_near[view:view+1],
                                                sample_far[view:view+1],
                                                truncation=self.opt.inference.truncation_ratio,
                                                truncation_latent=surface_mean_latent,
                                                return_sdf=True,
                                                return_xyz=True,
                                                exstyle=batch['exstyle'])
                    
                xyz = surface_out[2].cpu()
                sdf = surface_out[3].cpu()
                del surface_out
                torch.cuda.empty_cache()
                depth_mesh = xyz2mesh(xyz)
                mesh = Meshes(
                    verts=[torch.from_numpy(np.asarray(depth_mesh.vertices)).to(torch.float32).to(self.device)],
                    faces = [torch.from_numpy(np.asarray(depth_mesh.faces)).to(torch.float32).to(self.device)],
                    textures=None,
                    verts_normals=[torch.from_numpy(np.copy(np.asarray(depth_mesh.vertex_normals))).to(torch.float32).to(self.device)],
                )
                mesh = add_textures(mesh)
                cameras = create_cameras(azim=np.rad2deg(trajectory[view,0].cpu().numpy()),
                                                elev=np.rad2deg(trajectory[view,1].cpu().numpy()),
                                                fov=2*trajectory[view,2].cpu().numpy(),
                                                dist=1, device=self.device)
                renderer = create_mesh_renderer(cameras, image_size=512,
                                            light_location=((0.0,1.0,5.0),), specular_color=((0.2,0.2,0.2),),
                                            ambient_color=((0.1,0.1,0.1),), diffuse_color=((0.65,.65,.65),),
                                            device=self.device)

                mesh_image = 255 * renderer(mesh).cpu().numpy()
                mesh_image = mesh_image[...,:3]
                mesh_ls.append((mesh_image[0]).astype(np.uint8))
                torch.cuda.empty_cache()
                
            for k in range(num_frames):
                depth_writer.writeFrame(mesh_ls[k])
            depth_writer.close()

    def save_image(self, batch, forward_output, stem=None, save_dir=None, max_rows=11):
        downsampler_mode =self.opt.style.downsize_interpolation
        img_resized_ls = [F.interpolate(itm, size=(self.opt.style.vis_size, self.opt.style.vis_size), mode=downsampler_mode) for itm in forward_output]
        
        grid = make_grid(torch.cat(img_resized_ls, 0), nrow=min(max_rows,len(img_resized_ls)))
        if save_dir is None:
            parent_dir = os.path.join(self.opt.experiment.root_save_dir, 'vis', f'_{self.opt.experiment.exp_mode}')
            os.makedirs(parent_dir, exist_ok=True)
            save_img_dir = os.path.join(parent_dir, self.opt.style.jobname)
            os.makedirs(save_img_dir, exist_ok = True)
        else:
            save_img_dir = save_dir
        save_image(grid, '{}/{}.png'.format(save_img_dir, stem))
        print('image saved:','{}/{}.png'.format(save_img_dir, stem))
    
    def set_camera_trajectory(self):
            # set camera trajectory
        num_frames = self.opt.inference.num_frames
        # Generate video trajectory
        trajectory = np.zeros((num_frames,3), dtype=np.float32)
        # sweep azimuth angles (4 seconds)
        if self.opt.inference.azim_video:
            t = np.linspace(0, 1, num_frames)
            elev = 0
            fov = self.opt.camera.fov
            if self.opt.camera.uniform:
                azim = self.opt.camera.azim * np.cos(t * 2 * np.pi)
            else:
                azim = 1.5 * self.opt.camera.azim * np.cos(t * 2 * np.pi)

            trajectory[:num_frames,0] = azim
            trajectory[:num_frames,1] = elev
            trajectory[:num_frames,2] = fov

        # elipsoid sweep (4 seconds)
        else:
            t = np.linspace(0, 1, num_frames)
            fov = self.opt.camera.fov #+ 1 * np.sin(t * 2 * np.pi)
            if self.opt.camera.uniform:
                elev = self.opt.camera.elev / 2 + self.opt.camera.elev / 2  * np.sin(t * 2 * np.pi)
                azim = self.opt.camera.azim  * np.cos(t * 2 * np.pi)
            else:
                elev = 1.5 * self.opt.camera.elev * np.sin(t * 2 * np.pi)
                azim = 1.5 * self.opt.camera.azim * np.cos(t * 2 * np.pi)

            trajectory[:num_frames,0] = azim
            trajectory[:num_frames,1] = elev
            trajectory[:num_frames,2] = fov

        trajectory = torch.from_numpy(trajectory).to(self.device)
        return trajectory
    
    def image2_camsettings(self, input_img, is_thumb=False):
        if not is_thumb:
            thumb_img = self.pool_64(input_img)
        else:
            thumb_img = input_img

        assert thumb_img.shape[
            -1] == 64, 'check volume_discriminator input img dim'

        with torch.no_grad():
            _, pred_locations = self.volume_discriminator(thumb_img)
        pred_cam_settings = self._cam_locations_2_cam_settings(
            batch=input_img.shape[0], cam_locations=pred_locations)
        return pred_cam_settings

    def given_2camsettings(self):
        locations = torch.tensor([[self.opt.camera.given_azim, self.opt.camera.given_elev]], device=self.device)
        # fov = opt.camera.fov * torch.ones((locations.shape[0],1), device=self.device)
        cam = generate_camera_params(self.opt.model.renderer_spatial_output_dim, self.device, locations=locations[:,:2],
                            fov_ang=self.opt.camera.fov, dist_radius=self.opt.camera.dist_radius)
        return cam
    
    def _cam_locations_2_cam_settings(self, batch,
                                      cam_locations: torch.Tensor):
        device = self.device
        cam_settings = generate_camera_params(
            self.opt.training.renderer_output_size,
            device,
            batch,
            locations=cam_locations,
            #input_fov=fov,
            uniform=self.opt.camera.uniform,
            azim_range=self.opt.camera.azim,
            elev_range=self.opt.camera.elev,
            fov_ang=self.opt.camera.fov,
            dist_radius=self.opt.camera.dist_radius,)
            # return_calibs=True)
        return cam_settings    

    def load_edit_stats(self):
        self.ATTRS = ["Bangs", "Smiling", "No_Beard", "Young", "Eyeglasses"]
        self.boundaries = {k: {} for k in self.ATTRS}
        self.spaces = ('renderer', 'decoder')
        # self.spaces = ('renderer', )
        for attr_name in self.ATTRS:
            for space in self.spaces:
                boundary_file_path = os.path.join(self.opt.experiment.root_save_dir,'cache/editing_dirs/stylesdf',f'{space}_{attr_name}','boundary.npy')
                # boundary_file_path = boundary_root_path / 'boundaries_cvpr23' / 'stylesdf' / f'{space}_{attr_name}/boundary.npy'
                # boundary_file_path = boundary_root_path / 'stylesdf' / f'{space}_{attr_name}/boundary.npy'
                boundary = np.load(boundary_file_path)
                self.boundaries[attr_name][space] = boundary
                del boundary
        print('init editing directions done.')
            
    def edit_code(self, pred_latents, editing_boundary_scale_list=None):
        # ATTRS = self.ATTRS

        if editing_boundary_scale_list is None:
            editing_boundary_scale_list = [0, 1, 0, 0,
                                           0]  # add smile by default

        editing_boundary_scale_dict = dict(  # add smile by default
            Bangs=editing_boundary_scale_list[0],
            Smiling=editing_boundary_scale_list[1],
            No_Beard=editing_boundary_scale_list[2],
            Young=editing_boundary_scale_list[3],
            Eyeglasses=editing_boundary_scale_list[4])

        edited_codes = []

        spaces = ('renderer', 'decoder')

        for idx in range(2):
            space = spaces[idx]
            pred_latent_code = pred_latents[idx]
            new_codes = pred_latent_code.cpu().numpy()

            for i, attr_name in enumerate(self.ATTRS):
                boundary = self.boundaries[attr_name][space]

                if new_codes.ndim == 3:
                    boundary = np.expand_dims(
                        boundary,
                        1)  # 1 1 256, for broadcasting add with B 9 256
                new_codes += boundary * editing_boundary_scale_dict[attr_name]

            edited_codes.append(new_codes)

        edited_pred_latents = [
            torch.Tensor(code).to(self.device) for code in edited_codes
        ]
        # output_img_name = f"Bangs{editing_boundary_scale_dict['Bangs']}_Smile{editing_boundary_scale_dict['Smiling']}_Beard{editing_boundary_scale_dict['No_Beard']}.png"
        output_img_name = f"Bangs{editing_boundary_scale_dict['Bangs']}_Smile{editing_boundary_scale_dict['Smiling']}_Beard{editing_boundary_scale_dict['No_Beard']}_Young{editing_boundary_scale_dict['Young']}_Glass{editing_boundary_scale_dict['Eyeglasses']}"

        return dict(output_img_name=output_img_name,
                    edited_pred_latents=edited_pred_latents)

    def _truncate_w(self, w_hat, w_styles):
        # ref to: https://github.com/NIRVANALAN/ICCV23-3DMM-StyleSDF/blob/
        # add_3dmm_align/project/Runners/BasicModelAligner/mlp_based_aligner.py 
        """w_hat: 3DMM inferenced styles
        w_styles: original B, 256 w_styles
        """
        w_truncate_layer = 2 # self.opt.deep3DFaceRecon.w_truncate_layer

        if w_hat.ndim == 2:
            w_hat = w_hat.unsqueeze(1).repeat_interleave(w_truncate_layer, dim=1)
        else:
            assert w_hat.shape[1] == w_truncate_layer

        if w_styles.ndim == 2:
            w_styles = w_styles.unsqueeze(1).repeat_interleave(9-w_truncate_layer, dim=1)
        
        truncated_w = torch.cat((w_hat, w_styles), dim=1) # B, 9, 256 

        return truncated_w

if __name__ == '__main__':
    print(f'Hello DeformToon3D!')
    opt = BaseOptions().parse()
    opt.training.camera = opt.camera
    opt.training.size = opt.model.size
    opt.training.renderer_output_size = opt.model.renderer_spatial_output_dim
    opt.training.style_dim = opt.model.style_dim
    opt.model.freeze_renderer = False
    opt.rendering.offset_sampling = True
    opt.rendering.static_viewdirs = True
    opt.rendering.force_background = True
    if not opt.experiment.exp_mode == 'train':
        opt.rendering.perturb = 0 
    opt.inference.project_noise = opt.model.project_noise 
    opt.inference.return_xyz = opt.rendering.return_xyz 
    
    runner = Runner(opt)
    if opt.experiment.exp_mode == 'train':
        runner.train()
    elif opt.experiment.exp_mode == 'visualize_video':
        runner.visualize_video()
    elif opt.experiment.exp_mode == 'visualize_surface':
        runner.visualize_surface()