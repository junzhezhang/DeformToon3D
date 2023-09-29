import kornia
from pdb import set_trace as st
from torch.nn import functional as F
import numpy as np
import torch

from . import *


class E3DGELossClass(torch.nn.Module):
    def __init__(self, device, opt) -> None:
        super().__init__()

        self.opt = opt
        self.device = device
        self.criterionImg = torch.nn.MSELoss()
        self.criterionLPIPS = LPIPS(net_type='alex', device=device).eval()
        if opt.id_lambda > 0:
            self.criterionID = IDLoss(device=device).eval()
        self.id_loss_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

        # define 3d rec loss, for occupancy
        self.criterion3d_rec = torch.nn.SmoothL1Loss()

        print('init loss class finished', flush=True)

    def psnr_loss(self, input, target, max_val):
        return kornia.metrics.psnr(input, target, max_val)

    def calc_shape_rec_loss(self,
                            pred_shape: dict,
                            gt_shape: dict,
                            device,):
        """apply 3d shape reconstruction supervision. Basically supervise the densities with L1 loss

        Args:
            pred_shape (dict): dict contains reconstructed shape information
            gt_shape (dict): dict contains gt shape information
            supervise_sdf (bool, optional): whether supervise sdf rec. Defaults to True.
            supervise_surface_normal (bool, optional): whether supervise surface rec. Defaults to False.

        Returns:
            dict: shape reconstruction loss
        """

        shape_loss_dict = {}
        shape_loss = 0
        # assert supervise_sdf or supervise_surface_normal, 'should at least supervise one types of shape reconstruction'
        # todo, add weights

        if self.opt.shape_uniform_lambda > 0:
            shape_loss_dict['coarse'] = self.criterion3d_rec(
                pred_shape['coarse_densities'].squeeze(),
                gt_shape['coarse_densities'].squeeze(
                )) 
            shape_loss += shape_loss_dict['coarse'] * self.opt.shape_uniform_lambda

        if self.opt.shape_importance_lambda > 0:
            shape_loss_dict['fine'] = self.criterion3d_rec(
                pred_shape['fine_densities'].squeeze(), # ? how to supervise
                    gt_shape['fine_densities'].squeeze()) 
            shape_loss += shape_loss_dict['fine'] * self.opt.shape_importance_lambda 

        # TODO, add on surface pts supervision ?

        return shape_loss, shape_loss_dict

    def calc_2d_rec_loss(self,
                         input,
                         gt):
        opt = self.opt
        loss_dict = {}

        rec_loss = self.criterionImg(input, gt)
        lpips_loss = self.criterionLPIPS(input, gt)
        loss_psnr = self.psnr_loss((input / 2 + 0.5), (gt / 2 + 0.5),
                                   1.0)


        if opt.id_lambda > 0:
            if input.shape[-1] != 256:
                arcface_input = self.id_loss_pool(input)
                id_loss_gt = self.id_loss_pool(gt)
            else:
                arcface_input = input
                id_loss_gt = gt

            loss_id, _, _ = self.criterionID(arcface_input, id_loss_gt,
                                             id_loss_gt)
        else:
            loss_id = torch.tensor(0., device=input.device)

        loss = rec_loss * opt.l2_lambda + lpips_loss * opt.lpips_lambda + loss_id * opt.id_lambda

        # loss_ssim
        loss_ssim = kornia.losses.ssim_loss(input, gt, 5)  #?

        # if return_dict:
        loss_dict['loss_l2'] = rec_loss
        loss_dict['loss_id'] = loss_id
        loss_dict['loss_lpips'] = lpips_loss
        loss_dict['loss'] = loss

        # metrics to report, not involved in training
        loss_dict['mae'] = F.l1_loss(input, gt)
        loss_dict['PSNR'] = loss_psnr
        loss_dict['SSIM'] = 1 - loss_ssim  # Todo
        loss_dict['ID_SIM'] = 1 - loss_id

        return loss, loss_dict

        # return loss, rec_loss, lpips_loss, loss_id, 0, 0

    def forward(self, *args, **kwargs):

        return self.calc_2d_rec_loss(*args, **kwargs)
