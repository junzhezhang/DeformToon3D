import os
import numpy as np
import torch
from util import save_image, load_image
import argparse
from argparse import Namespace
from torchvision import transforms
from torch.nn import functional as F
import torchvision
from model.dualstylegan import DualStyleGAN
from model.encoder.psp import pSp

class TestOptions():
    def __init__(self):

        self.parser = argparse.ArgumentParser(description="Exemplar-Based Style Transfer")
        self.parser.add_argument("--content", type=str, default='./data/content/081680.jpg', help="path of the content image")
        self.parser.add_argument("--style", type=str, default='cartoon', help="target style type")
        self.parser.add_argument("--style_id", type=int, default=53, help="the id of the style image")
        self.parser.add_argument("--truncation", type=float, default=0.75, help="truncation for intrinsic style code (content)")
        self.parser.add_argument("--weight", type=float, nargs=18, default=[0.75]*7+[1]*11, help="weight of the extrinsic style")
        self.parser.add_argument("--name", type=str, default='cartoon_transfer', help="filename to save the generated images")
        self.parser.add_argument("--preserve_color", action="store_true", help="preserve the color of the content image")
        self.parser.add_argument("--model_path", type=str, default='./checkpoint/', help="path of the saved models")
        self.parser.add_argument("--model_name", type=str, default='generator.pt', help="name of the saved dualstylegan")
        self.parser.add_argument("--output_path", type=str, default='./output/', help="path of the output images")
        self.parser.add_argument("--data_path", type=str, default='./data/', help="path of dataset")
        self.parser.add_argument("--align_face", action="store_true", help="apply face alignment to the content image")
        self.parser.add_argument("--exstyle_name", type=str, default=None, help="name of the extrinsic style codes")
        self.parser.add_argument("--wplus", action="store_true", help="use original pSp encoder to extract the intrinsic style code")
        self.parser.add_argument("--start_from", type=int, default=0,  help="starting from which")
        self.parser.add_argument("--end_until", type=int, default=1e9,  help="end until which")
        self.parser.add_argument("--save_overview", action="store_true", help="preserve the color of the content image")

    def parse(self):
        self.opt = self.parser.parse_args()
        if self.opt.exstyle_name is None:
            if os.path.exists(os.path.join(self.opt.model_path, self.opt.style, 'refined_exstyle_code.npy')):
                self.opt.exstyle_name = 'refined_exstyle_code.npy'
            else:
                self.opt.exstyle_name = 'exstyle_code.npy'        
        args = vars(self.opt)
        print('Load options')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt
    
def run_alignment(args):
    import dlib
    from model.encoder.align_all_parallel import align_face
    modelname = os.path.join(args.model_path, 'shape_predictor_68_face_landmarks.dat')
    if not os.path.exists(modelname):
        import wget, bz2
        wget.download('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2', modelname+'.bz2')
        zipfile = bz2.BZ2File(modelname+'.bz2')
        data = zipfile.read()
        open(modelname, 'wb').write(data) 
    predictor = dlib.shape_predictor(modelname)
    aligned_image = align_face(filepath=args.content, predictor=predictor)
    return aligned_image


if __name__ == "__main__":
    device = "cuda"

    parser = TestOptions()
    args = parser.parse()
    print('*'*98)
    
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
    ])
    
    generator = DualStyleGAN(1024, 512, 8, 2, res_index=6)
    generator.eval()

    ckpt = torch.load(os.path.join(args.model_path, args.style, args.model_name), map_location=lambda storage, loc: storage)
    generator.load_state_dict(ckpt["g_ema"])
    generator = generator.to(device)
    
    if args.wplus:
        model_path = os.path.join(args.model_path, 'encoder_wplus.pt')
    else:
        model_path = os.path.join(args.model_path, 'encoder.pt')
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    if 'output_size' not in opts:
        opts['output_size'] = 1024    
    opts = Namespace(**opts)
    opts.device = device
    encoder = pSp(opts)
    encoder.eval()
    encoder.to(device)

    exstyles = np.load(os.path.join(args.model_path, args.style, args.exstyle_name), allow_pickle='TRUE').item()

    z_plus_latent=not args.wplus
    return_z_plus_latent=not args.wplus
    input_is_latent=args.wplus    
    
    print('Load models successfully!')
    import glob
    import cv2
    pathnames = sorted(glob.glob(args.content+'/*fullhd*'))
    print('To process {} images'.format(len(pathnames)))
    import facexlib.utils.face_restoration_helper as face_restoration_helper
    from facexlib.alignment import init_alignment_model, landmark_98_to_68

    align_net = init_alignment_model('awing_fan') # TODO verify the model
    FaceRestoreHelper = face_restoration_helper.FaceRestoreHelper(
                        upscale_factor=1, face_size=256)


    
    for i, filename in enumerate(pathnames):
        if i < args.start_from or i >= args.end_until:
            continue
        if i%100 == 0:
            print('processing {}th image'.format(i))

        with torch.no_grad():
            viz = []
            # load content image
            if args.align_face:
                # raise NotImplementedError
                # I = transform(run_alignment(args)).unsqueeze(dim=0).to(device)
                # I = F.adaptive_avg_pool2d(I, 1024)
                FaceRestoreHelper.clean_all()
                FaceRestoreHelper.read_image(filename)
            
                FaceRestoreHelper.get_face_landmarks_5()
                # I = FaceRestoreHelper.align_warp_face()[..., ::-1]/ 255 # (256, 256, 3)
                I = FaceRestoreHelper.align_warp_face()/255
                
                # print('shape after align_face():', np.array(I).shape)
                
                I = transform(I).unsqueeze(dim=0).to(device)
                I = I.float() # float64 -> float32
            else:
                I = load_image(filename).to(device)
                if I.shape[-1] == 512:
                    I = F.interpolate(I, scale_factor=2, mode='bicubic', align_corners=False)
                    # I = cv2.resize(I, (1024,1024))
                    input_res = 512
                else:
                    input_res = 1024
            viz += [I]

            # reconstructed content image and its intrinsic style code
            img_rec, instyle = encoder(F.adaptive_avg_pool2d(I, 256), randomize_noise=False, return_latents=True, 
                                    z_plus_latent=z_plus_latent, return_z_plus_latent=return_z_plus_latent, resize=False)  
            img_rec = torch.clamp(img_rec.detach(), -1, 1)
            viz += [img_rec]

            stylename = list(exstyles.keys())[args.style_id]
            latent = torch.tensor(exstyles[stylename]).to(device)
            if args.preserve_color and not args.wplus:
                latent[:,7:18] = instyle[:,7:18]
            # extrinsic styte code
            exstyle = generator.generator.style(latent.reshape(latent.shape[0]*latent.shape[1], latent.shape[2])).reshape(latent.shape)
            if args.preserve_color and args.wplus:
                exstyle[:,7:18] = instyle[:,7:18]
                
            # load style image if it exists
            S = None
            if os.path.exists(os.path.join(args.data_path, args.style, 'images/train', stylename)):
                S = load_image(os.path.join(args.data_path, args.style, 'images/train', stylename)).to(device)
                viz += [S]

            # style transfer 
            # input_is_latent: instyle is not in W space
            # z_plus_latent: instyle is in Z+ space
            # use_res: use extrinsic style path, or the style is not transferred
            # interp_weights: weight vector for style combination of two paths
            img_gen, _ = generator([instyle], exstyle, input_is_latent=input_is_latent, z_plus_latent=z_plus_latent,
                                truncation=args.truncation, truncation_latent=0, use_res=True, interp_weights=args.weight)
            img_gen = torch.clamp(img_gen.detach(), -1, 1)
            viz += [img_gen]

        # print('Generate images successfully!')
        # import pdb; pdb.set_trace()
        
        # save_name = args.name+'_%d_%s'%(args.style_id, os.path.basename(args.content).split('.')[0])
        # save_name = str(i)
        save_name = os.path.basename(filename).split('.')[0]
        os.makedirs(args.output_path, exist_ok=True)
        if input_res == 512:
            # bilinear or bicubic
            img_gen = F.interpolate(img_gen, scale_factor=0.5, mode='bilinear', align_corners=False)

        if args.save_overview:
            viz_resize = [F.adaptive_avg_pool2d(itm, 256) for itm in viz]
            save_image(torchvision.utils.make_grid((torch.cat(viz_resize, dim=0)), 4, 2).cpu(), 
                    os.path.join(args.output_path, save_name+'_overview.jpg'))
        save_image(img_gen[0].cpu(), os.path.join(args.output_path, save_name+'.png'))

        print(f'Save {filename} successfully!')
