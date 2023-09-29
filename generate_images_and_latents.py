import os
import torch
import trimesh
import numpy as np
from munch import *
from PIL import Image
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils import data
from torchvision import utils
from torchvision import transforms
from skimage.measure import marching_cubes
from scipy.spatial import Delaunay
from options import BaseOptions
from model import Generator
from utils import (
    generate_camera_params,
    align_volume,
    extract_mesh_with_marching_cubes,
    xyz2mesh,
)


torch.random.manual_seed(1234)


def generate(opt, g_ema, surface_g_ema, device, mean_latent, surface_mean_latent):
    g_ema.eval()
    if not opt.no_surface_renderings:
        surface_g_ema.eval()

    # set camera angles
    if opt.fixed_camera_angles:
        # These can be changed to any other specific viewpoints.
        # You can add or remove viewpoints as you wish
        locations = torch.tensor([[0, 0],
                                  [-1.5 * opt.camera.azim, 0],
                                  [-1 * opt.camera.azim, 0],
                                  [-0.5 * opt.camera.azim, 0],
                                  [0.5 * opt.camera.azim, 0],
                                  [1 * opt.camera.azim, 0],
                                  [1.5 * opt.camera.azim, 0],
                                  [0, -1.5 * opt.camera.elev],
                                  [0, -1 * opt.camera.elev],
                                  [0, -0.5 * opt.camera.elev],
                                  [0, 0.5 * opt.camera.elev],
                                  [0, 1 * opt.camera.elev],
                                  [0, 1.5 * opt.camera.elev]], device=device)
        # For zooming in/out change the values of fov
        # (This can be defined for each view separately via a custom tensor
        # like the locations tensor above. Tensor shape should be [locations.shape[0],1])
        # reasonable values are [0.75 * opt.camera.fov, 1.25 * opt.camera.fov]
        fov = opt.camera.fov * torch.ones((locations.shape[0],1), device=device)
        num_viewdirs = locations.shape[0]
    else: # draw random camera angles
        locations = None
        # fov = None
        fov = opt.camera.fov
        num_viewdirs = opt.num_views_per_id

    dict_to_save = {}
    opt.results_dst_dir = os.path.join(opt.output_root_dir, 'real_space')
    latent_filepath = os.path.join(opt.output_root_dir, f'latents.pth')
    os.makedirs(opt.output_root_dir, exist_ok=True)
    os.makedirs(opt.results_dst_dir, exist_ok=True)
    # generate images
    for i in tqdm(range(opt.identities)):  
        with torch.no_grad():
            chunk = 8
            sample_z = torch.randn(1, opt.style_dim, device=device).repeat(num_viewdirs,1)
            sample_cam_extrinsics, sample_focals, sample_near, sample_far, sample_locations = \
            generate_camera_params(opt.renderer_output_size, device, batch=num_viewdirs,
                                   locations=locations, #input_fov=fov,
                                   uniform=opt.camera.uniform, azim_range=opt.camera.azim,
                                   elev_range=opt.camera.elev, fov_ang=fov,
                                   dist_radius=opt.camera.dist_radius)

            dict_to_save['{}'.format(str(i).zfill(7))] = {
                'z': sample_z.detach().cpu()[0],
                'cam_extrinsics': sample_cam_extrinsics.detach().cpu()[0],
                'focal': sample_focals.detach().cpu()[0],
                'near': sample_near.detach().cpu()[0],
                'far': sample_far.detach().cpu()[0],
                'location': sample_locations.detach().cpu()[0]
            }
            
            rgb_images = torch.Tensor(0, 3, opt.size, opt.size)
            rgb_images_thumbs = torch.Tensor(0, 3, opt.renderer_output_size, opt.renderer_output_size)
            for j in range(0, num_viewdirs, chunk):
                out = g_ema([sample_z[j:j+chunk]],
                            sample_cam_extrinsics[j:j+chunk],
                            sample_focals[j:j+chunk],
                            sample_near[j:j+chunk],
                            sample_far[j:j+chunk],
                            truncation=opt.truncation_ratio,
                            truncation_latent=mean_latent)
                
                rgb_images = torch.cat([rgb_images, out[0].cpu()], 0)
                rgb_images_thumbs = torch.cat([rgb_images_thumbs, out[1].cpu()], 0)
            utils.save_image(rgb_images,
                os.path.join(opt.results_dst_dir,'{}_fullhd.png'.format(str(i).zfill(7))),
                nrow=num_viewdirs,
                normalize=True,
                padding=0,
                value_range=(-1, 1),)

            # this is done to fit to RTX2080 RAM size (11GB)
            del out
            torch.cuda.empty_cache()

    torch.save(dict_to_save, latent_filepath)
    print('Saved latent vectors to {}'.format(latent_filepath))

if __name__ == "__main__":
    device = "cuda"
    opt = BaseOptions().parse()
    opt.model.is_test = True
    opt.model.freeze_renderer = False
    opt.rendering.offset_sampling = True
    opt.rendering.static_viewdirs = True
    opt.rendering.force_background = True
    opt.rendering.perturb = 0
    opt.inference.size = opt.model.size
    opt.inference.camera = opt.camera
    opt.inference.renderer_output_size = opt.model.renderer_spatial_output_dim
    opt.inference.style_dim = opt.model.style_dim
    opt.inference.project_noise = opt.model.project_noise
    opt.inference.return_xyz = opt.rendering.return_xyz
    # find checkpoint directory
    # check if there's a fully trained model
    checkpoints_dir = 'full_models'
    checkpoint_path = os.path.join(checkpoints_dir, opt.experiment.expname + '.pt')
    if os.path.isfile(checkpoint_path):
        # define results directory name
        result_model_dir = 'final_model'
    else:
        checkpoints_dir = os.path.join('checkpoint', opt.experiment.expname, 'full_pipeline')
        checkpoint_path = os.path.join(checkpoints_dir,
                                       'models_{}.pt'.format(opt.experiment.ckpt.zfill(7)))
        # define results directory name
        result_model_dir = 'iter_{}'.format(opt.experiment.ckpt.zfill(7))

    # create results directory
    results_dir_basename = os.path.join(opt.inference.results_dir, opt.experiment.expname)
    opt.inference.results_dst_dir = os.path.join(results_dir_basename, result_model_dir)
    if opt.inference.fixed_camera_angles:
        opt.inference.results_dst_dir = os.path.join(opt.inference.results_dst_dir, 'fixed_angles')
    else:
        opt.inference.results_dst_dir = os.path.join(opt.inference.results_dst_dir, 'random_angles')
    os.makedirs(opt.inference.results_dst_dir, exist_ok=True)
    os.makedirs(os.path.join(opt.inference.results_dst_dir, 'images'), exist_ok=True)
    if not opt.inference.no_surface_renderings:
        os.makedirs(os.path.join(opt.inference.results_dst_dir, 'depth_map_meshes'), exist_ok=True)
        os.makedirs(os.path.join(opt.inference.results_dst_dir, 'marching_cubes_meshes'), exist_ok=True)

    # load saved model
    checkpoint = torch.load(checkpoint_path)

    # load image generation model
    g_ema = Generator(opt.model, opt.rendering, opt.style).to(device)
    pretrained_weights_dict = checkpoint["g_ema"]
    model_dict = g_ema.state_dict()
    for k, v in pretrained_weights_dict.items():
        if v.size() == model_dict[k].size():
            model_dict[k] = v

    g_ema.load_state_dict(model_dict)

    # load a second volume renderer that extracts surfaces at 128x128x128 (or higher) for better surface resolution
    if not opt.inference.no_surface_renderings:
        opt['surf_extraction'] = Munch()
        opt.surf_extraction.rendering = opt.rendering
        opt.surf_extraction.model = opt.model.copy()
        opt.surf_extraction.model.renderer_spatial_output_dim = 128
        opt.surf_extraction.rendering.N_samples = opt.surf_extraction.model.renderer_spatial_output_dim
        opt.surf_extraction.rendering.return_xyz = True
        opt.surf_extraction.rendering.return_sdf = True
        surface_g_ema = Generator(opt.surf_extraction.model, opt.surf_extraction.rendering, opt.style, full_pipeline=False).to(device)

        # Load weights to surface extractor
        surface_extractor_dict = surface_g_ema.state_dict()
        for k, v in pretrained_weights_dict.items():
            if k in surface_extractor_dict.keys() and v.size() == surface_extractor_dict[k].size():
                surface_extractor_dict[k] = v

        surface_g_ema.load_state_dict(surface_extractor_dict)
    else:
        surface_g_ema = None

    # get the mean latent vector for g_ema
    if opt.inference.truncation_ratio < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(opt.inference.truncation_mean, device)
    else:
        surface_mean_latent = None

    # get the mean latent vector for surface_g_ema
    if not opt.inference.no_surface_renderings:
        surface_mean_latent = mean_latent[0]
    else:
        surface_mean_latent = None

    generate(opt.inference, g_ema, surface_g_ema, device, mean_latent, surface_mean_latent)
