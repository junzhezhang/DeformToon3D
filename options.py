import configargparse
from munch import *
from pdb import set_trace as st

class BaseOptions():
    def __init__(self):
        self.parser = configargparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # Dataset options
        dataset = self.parser.add_argument_group('dataset')
        dataset.add_argument("--dataset_path", type=str, default='./datasets/FFHQ', help="path to the lmdb dataset")

        # Experiment Options
        experiment = self.parser.add_argument_group('experiment')
        experiment.add_argument('--config', is_config_file=True, help='config file path')
        experiment.add_argument("--expname", type=str, default='ffhq1024x1024', help='experiment name')
        experiment.add_argument("--ckpt", type=str, default='300000', help="path to the checkpoints to resume training")
        experiment.add_argument("--continue_training", type=int, default='-1', help="continue training the model, no condition training if -1")
        experiment.add_argument("--load_epoch", type=str, default='latest', help='load epoch')
        experiment.add_argument("--root_save_dir", type=str, default='.', help="if ., use the same path as the code, otherwise, use the specified path")
        experiment.add_argument("--exp_mode", type=str, default='train', help="train|etc")
        
        # Training loop options
        training = self.parser.add_argument_group('training')
        training.add_argument("--checkpoints_dir", type=str, default='./checkpoint', help='checkpoints directory name')
        training.add_argument("--iter", type=int, default=300000, help="total number of training iterations")
        training.add_argument("--batch", type=int, default=4, help="batch sizes for each GPU. A single RTX2080 can fit batch=4, chunck=1 into memory.")
        training.add_argument("--chunk", type=int, default=4, help='number of samples within a batch to processed in parallel, decrease if running out of memory')
        training.add_argument("--val_n_sample", type=int, default=8, help="number of test samples generated during training")
        training.add_argument("--d_reg_every", type=int, default=16, help="interval for applying r1 regularization to the StyleGAN generator")
        training.add_argument("--g_reg_every", type=int, default=4, help="interval for applying path length regularization to the StyleGAN generator")
        training.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
        training.add_argument("--mixing", type=float, default=0.9, help="probability of latent code mixing")
        training.add_argument("--lr", type=float, default=0.002, help="learning rate")
        training.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
        training.add_argument("--view_lambda", type=float, default=15, help="weight of the viewpoint regularization")
        training.add_argument("--eikonal_lambda", type=float, default=0.1, help="weight of the eikonal regularization")
        training.add_argument("--min_surf_lambda", type=float, default=0.05, help="weight of the minimal surface regularization")
        training.add_argument("--min_surf_beta", type=float, default=100.0, help="weight of the minimal surface regularization")
        training.add_argument("--path_regularize", type=float, default=2, help="weight of the path length regularization")
        training.add_argument("--path_batch_shrink", type=int, default=2, help="batch size reducing factor for the path length regularization (reduce memory consumption)")
        training.add_argument("--wandb", action="store_true", help="use weights and biases logging")
        training.add_argument("--no_sphere_init", action="store_true", help="do not initialize the volume renderer with a sphere SDF")
        # training.add_argument('--find_unused_parameters', action="store_true", help='Distributed training, if True error for full_pipeline')

        # Inference Options
        inference = self.parser.add_argument_group('inference')
        inference.add_argument("--results_dir", type=str, default='./evaluations', help='results/evaluations directory name')
        inference.add_argument("--truncation_ratio", type=float, default=0.5, help="truncation ratio, controls the diversity vs. quality tradeoff. Higher truncation ratio would generate more diverse results")
        inference.add_argument("--truncation_mean", type=int, default=10000, help="number of vectors to calculate mean for the truncation")
        inference.add_argument("--identities", type=int, default=16, help="number of identities to be generated")
        inference.add_argument("--num_views_per_id", type=int, default=1, help="number of viewpoints generated per identity")
        inference.add_argument("--no_surface_renderings", action="store_true", help="when true, only RGB outputs will be generated. otherwise, both RGB and depth videos/renderings will be generated. this cuts the processing time per video")
        inference.add_argument("--fixed_camera_angles", action="store_true", help="when true, the generator will render indentities from a fixed set of camera angles.")
        inference.add_argument("--azim_video", action="store_true", help="when true, the camera trajectory will travel along the azimuth direction. Otherwise, the camera will travel along an ellipsoid trajectory.")
        inference.add_argument("--output_root_dir", type=str, default='data/real_space', help="output dir for generate training images")
        inference.add_argument("--from_which_subject", type=int, default=1000, help="eval from which subject if given latent")
        inference.add_argument("--num_frames", type=int, default=250, help="num of frames for render videos")
        inference.add_argument("--render_subjects_wt_style_gt", action="store_true", help="if visual vtoonify fail subjects")
        inference.add_argument("--given_subject_list", type=str, default='1000-1100', help='condition on feat layers, if got multiple, concat')
        inference.add_argument("--GAN_inversion", action="store_true", help="when true, get latent and cam from inversion")
        inference.add_argument("--inversion_latent_path", type=str, default='/mnt/e/Downloads/FFHQ_new/ffhq-dataset/real-case-latents/celeba-hq/w-space/pred_latents.pt', help="path")
        inference.add_argument("--inversion_image_path", type=str, default='', help="path")
        inference.add_argument("--inversion_latent_type", type=str, default='z', help="if empty, will give error, z|w|w+|w+half_0|w+half_8")
        inference.add_argument("--input_is_latent", action="store_true", help="when true W or W+ space inversion")
        inference.add_argument("--base_model_path", type=str, default='ffhq1024x1024.pt', help="base model for interpolation")
        inference.add_argument("--interpolation_mode", type=str, default='full', help="interpolation mode for baseline method, full|renderer|decoder")
        inference.add_argument("--checkpoint_path", type=str, default='', help='checkpoint for baseline')        
        inference.add_argument("--given_latent_cam", action="store_true")  

        # Generator options
        model = self.parser.add_argument_group('model')
        model.add_argument("--size", type=int, default=1024, help="image sizes for the model")
        model.add_argument("--style_dim", type=int, default=256, help="number of style input dimensions")
        model.add_argument("--channel_multiplier", type=int, default=2, help="channel multiplier factor for the StyleGAN decoder. config-f = 2, else = 1")
        model.add_argument("--n_mlp", type=int, default=8, help="number of mlp layers in stylegan's mapping network")
        model.add_argument("--lr_mapping", type=float, default=0.01, help='learning rate reduction for mapping network MLP layers')
        model.add_argument("--renderer_spatial_output_dim", type=int, default=64, help='spatial resolution of the StyleGAN decoder inputs')
        model.add_argument("--project_noise", action='store_true', help='when true, use geometry-aware noise projection to reduce flickering effects (see supplementary section C.1 in the paper). warning: processing time significantly increases with this flag to ~20 minutes per video.')

        # Camera options
        camera = self.parser.add_argument_group('camera')
        camera.add_argument("--uniform", action="store_true", help="when true, the camera position is sampled from uniform distribution. Gaussian distribution is the default")
        camera.add_argument("--azim", type=float, default=0.3, help="camera azimuth angle std/range in Radians")
        camera.add_argument("--elev", type=float, default=0.15, help="camera elevation angle std/range in Radians")
        camera.add_argument("--fov", type=float, default=6, help="camera field of view half angle in Degrees")
        camera.add_argument("--dist_radius", type=float, default=0.12, help="radius of points sampling distance from the origin. determines the near and far fields")
        camera.add_argument("--given_azim", type=float, default=-10, help="camera azimuth angle std/range in Radians")
        camera.add_argument("--given_elev", type=float, default=-10, help="camera elevation angle std/range in Radians")

        # Volume Renderer options
        rendering = self.parser.add_argument_group('rendering')
        # MLP model parameters
        rendering.add_argument("--depth", type=int, default=8, help='layers in network')
        rendering.add_argument("--width", type=int, default=256, help='channels per layer')
        # Volume representation options
        rendering.add_argument("--no_sdf", action='store_true', help='By default, the raw MLP outputs represent an underline signed distance field (SDF). When true, the MLP outputs represent the traditional NeRF density field.')
        rendering.add_argument("--no_z_normalize", action='store_true', help='By default, the model normalizes input coordinates such that the z coordinate is in [-1,1]. When true that feature is disabled.')
        rendering.add_argument("--static_viewdirs", action='store_true', help='when true, use static viewing direction input to the MLP')
        # Ray intergration options
        rendering.add_argument("--N_samples", type=int, default=24, help='number of samples per ray')
        rendering.add_argument("--no_offset_sampling", action='store_true', help='when true, use random stratified sampling when rendering the volume, otherwise offset sampling is used. (See Equation (3) in Sec. 3.2 of the paper)')
        rendering.add_argument("--perturb", type=float, default=1., help='set to 0. for no jitter, 1. for jitter')
        rendering.add_argument("--raw_noise_std", type=float, default=0., help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
        rendering.add_argument("--force_background", action='store_true', help='force the last depth sample to act as background in case of a transparent ray')
        # Set volume renderer outputs
        rendering.add_argument("--return_xyz", action='store_true', help='when true, the volume renderer also returns the xyz point could of the surface. This point cloud is used to produce depth map renderings')
        rendering.add_argument("--return_sdf", action='store_true', help='when true, the volume renderer also returns the SDF network outputs for each location in the volume')
        
        # Style Transfer options
        style = self.parser.add_argument_group('style')
        # general for training
        style.add_argument("--num_workers", type=int, default=16, 
                           help="num of workers for dataloader")
        style.add_argument("--style_batch", type=int, default=16, 
                           help="batch size")
        style.add_argument("--style_chunk", type=int, default=1, 
                           help='number of samples within a batch to processed in parallel, decrease if running out of memory')
        style.add_argument("--n_epoch", type=int, default=100, 
                           help="n epoch")
        style.add_argument("--train_lr", type=float, default=5e-4, 
                           help='training learning rate')
        style.add_argument("--d_lr", type=float, default=2e-4, 
                           help='discriminator learning rate')
        style.add_argument("--train_split_ratio", type=float, default=1, 
                           help="train split ratio, it is 1 default, as all are synthetic 2D data.") 
        style.add_argument("--data_dir", type=str, default='./data', 
                           help='data directory')
        style.add_argument("--source_data", type=str, default='real_space', 
                           help='source data')
        style.add_argument("--style_data", type=str, 
                           default='real_space+style_pixar_9_0705+style_comic_34_0707+style_slamdunk_66_0607+style_caricature_17_0808+style_caricature_49_0808+style_caricature_92_0808+style_cartoon_91_0805+style_cartoon_299_0505+style_cartoon_221_0710+style_cartoon_252_0808', 
                           help='style data')
        style.add_argument("--latents_eval", type=str, 
                           default='latents_eval.pth', 
                           help='pth file of latents for eval')
        style.add_argument("--weight_decay", type=float, default=0, 
                           help='weight_decay')
        style.add_argument("--jobname", type=str, default='dummy', 
                           help='jobname')
        # checkpoint and visualization
        style.add_argument("--log_interval", type=int, default=10, 
                           help="log interval for display train loss")
        style.add_argument("--save_checkpoint_interval", type=int, default=5, 
                           help="frequency for saving checkpoints")
        style.add_argument("--save_latest_checkpoint_interval", type=int, default=1, 
                           help="frequency for saving latest checkpoints with overwrite")
        style.add_argument("--vis_interval", type=int, default=4096, 
                           help="visualize every 4k images")
        style.add_argument("--vis_size", type=int, default=256, 
                           help='size for visualized images')
        style.add_argument("--downsize_interpolation", type=str, default='nearest', 
                           help='interpolation for downsampling images')  
        # losses
        style.add_argument("--l1_loss", type=float, default=0, \
                           help='Weight of L1 loss, disabled if 0.')
        style.add_argument("--percep_loss", type=float, default=1.0, 
                           help="Weight of perceputal loss, disabled if 0.")
        style.add_argument("--elastic_loss", type=float, default=0.01,
                           help="Weight of elastic loss, disabled if 0.")
        style.add_argument("--gan_loss", type=float, default=0, 
                           help="Weight of GAN loss, disabled if 0. It is set 0 at the beginning of training, and 0.05 after 50 epochs.")
        # style.add_argument("--gan_loss_from_n_epoch", type=float, default=50, 
        #                    help="Enable GAN loss from which epoch.")
        # StyleField
        style.add_argument("--style_field", action='store_false', default=True,
                           help='adaptive style mixing')
        style.add_argument("--style_field_option", type=str, default='SIREN', 
                           help='SIREN|MLP')
        style.add_argument("--style_field_depth", type=int, default=4, 
                           help="Depth of StyleField.")
        style.add_argument("--condition_latent_dim", type=int, default=256, 
                           help='dimension of latent code for condition')
        style.add_argument("--condition_on_nerf_feat", type=str, default='', 
                           help='conditioned on nerf features, 0,1,2, concat')
        # Adaptive Style Mixing
        style.add_argument("--adaptive_style_mixing", action='store_false', default=True,
                           help='adaptive style mixing')
        style.add_argument("--adaptive_style_mixing_blocks", type=str, default='conv4+conv3+conv2+conv1+conv0', 
                           help='adaptive style mixing blocks, conv4, conv3, conv2, conv1, conv0, renderer etc, connect with +')
        style.add_argument("--adaptive_style_mixing_block_depth", type=int, default=3, 
                           help='adaptive style mixing t_c block depth')
        style.add_argument("--n_dim_exstyle", type=int, default=256, 
                           help='Dimension of exstyle condition')
        # Inference and application related
        style.add_argument("--decoder_interp_weights", type=float, default=1, help='decoder_interp_weights, for control degree of styles')
        style.add_argument("--style_field_dx_scale", type=float, default=1, help='style_field_dx_scale for reference time')
        style.add_argument("--n_styles", type=int, default=1, help='n_styles, which can be updated by dataset class')
        style.add_argument("--style_id", type=int, default=0, help='')
        style.add_argument("--interp_mode", type=str, default='', help='both_decoder_dx_scale|dx_scale|decoder')
        style.add_argument("--interp_weights_str", type=str, default='0,0.5,0.75,0.9,1', help='both_decoder_dx_scale|dx_scale|decoder')



        self.initialized = True

    def parse(self):
        self.opt = Munch()
        if not self.initialized:
            self.initialize()
        try:
            args = self.parser.parse_args()
        except: # solves argparse error in google colab
            args = self.parser.parse_args(args=[])

        for group in self.parser._action_groups[2:]:
            title = group.title
            self.opt[title] = Munch()
            for action in group._group_actions:
                dest = action.dest
                self.opt[title][dest] = args.__getattribute__(dest)

        return self.opt
