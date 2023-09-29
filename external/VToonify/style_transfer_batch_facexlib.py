import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import argparse
import numpy as np
import cv2
import torch
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
from model.vtoonify import VToonify
from model.bisenet.model import BiSeNet
# from model.encoder.align_all_parallel import align_face
from util import save_image, load_image, visualize, load_psp_standalone, get_crop_parameter_given_landmarks, tensor2cv2
import glob
from termcolor import colored
import facexlib.utils.face_restoration_helper as face_restoration_helper
from facexlib.alignment import init_alignment_model, landmark_98_to_68
"""
This script is extended from style_transfer.py
"""

class TestOptions():
    def __init__(self):

        self.parser = argparse.ArgumentParser(description="Style Transfer")
        self.parser.add_argument("--content", type=str, default='./data/077436.jpg', help="path of the content image/video")
        self.parser.add_argument("--style_id", type=int, default=26, help="the id of the style image")
        self.parser.add_argument("--style_degree", type=float, default=0.5, help="style degree for VToonify-D")
        self.parser.add_argument("--color_transfer", action="store_true", help="transfer the color of the style")
        self.parser.add_argument("--ckpt", type=str, default='./checkpoint/vtoonify_d_cartoon/vtoonify_s_d.pt', help="path of the saved model")
        self.parser.add_argument("--output_path", type=str, default='./output/', help="path of the output images")
        self.parser.add_argument("--scale_image", action="store_true", help="resize and crop the image to best fit the model")
        self.parser.add_argument("--style_encoder_path", type=str, default='./checkpoint/encoder.pt', help="path of the style encoder")
        self.parser.add_argument("--exstyle_path", type=str, default=None, help="path of the extrinsic style code")
        self.parser.add_argument("--faceparsing_path", type=str, default='./checkpoint/faceparsing.pth', help="path of the face parsing model")
        self.parser.add_argument("--video", action="store_true", help="if true, video stylization; if false, image stylization")
        self.parser.add_argument("--cpu", action="store_true", help="if true, only use cpu")
        self.parser.add_argument("--backbone", type=str, default='dualstylegan', help="dualstylegan | toonify")
        self.parser.add_argument("--padding", type=int, nargs=4, default=[200,200,200,200], help="left, right, top, bottom paddings to the face center")
        self.parser.add_argument("--batch_size", type=int, default=4, help="batch size of frames when processing video")
        self.parser.add_argument("--parsing_map_path", type=str, default=None, help="path of the refined parsing map of the target video")
        # enable multi-processing
        self.parser.add_argument("--start_from", type=int, default=0,  help="starting from which")
        self.parser.add_argument("--end_until", type=int, default=1e9,  help="end until which")
        
    def parse(self):
        self.opt = self.parser.parse_args()
        if self.opt.exstyle_path is None:
            self.opt.exstyle_path = os.path.join(os.path.dirname(self.opt.ckpt), 'exstyle_code.npy')
        args = vars(self.opt)
        print('Load options')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt
    
if __name__ == "__main__":

    parser = TestOptions()
    args = parser.parse()
    print('*'*98)
    
    
    device = "cpu" if args.cpu else "cuda"
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
        ])
    
    vtoonify = VToonify(backbone = args.backbone)
    vtoonify.load_state_dict(torch.load(args.ckpt, map_location=lambda storage, loc: storage)['g_ema'])
    vtoonify.to(device)

    parsingpredictor = BiSeNet(n_classes=19)
    parsingpredictor.load_state_dict(torch.load(args.faceparsing_path, map_location=lambda storage, loc: storage))
    parsingpredictor.to(device).eval()

    pspencoder = load_psp_standalone(args.style_encoder_path, device)    

    if args.backbone == 'dualstylegan':
        exstyles = np.load(args.exstyle_path, allow_pickle='TRUE').item()
        stylename = list(exstyles.keys())[args.style_id]
        exstyle = torch.tensor(exstyles[stylename]).to(device)
        with torch.no_grad():  
            exstyle = vtoonify.zplus2wplus(exstyle)

    if args.video and args.parsing_map_path is not None:
        x_p_hat = torch.tensor(np.load(args.parsing_map_path))          
            
    print('Load models successfully!')
    
    # pathnames = glob.glob(args.content+'/*') # NOTE for FFHQ
    pathnames = glob.glob(args.content+'/*fullhd*')
    pathnames = sorted([p for p in pathnames if 'vtoonify' not in p]) # sort and exclude toonified images in case process multiple rounds
    print('to process {} images'.format(len(pathnames)))
    
    align_net = init_alignment_model('awing_fan') # TODO verify the model
    FaceRestoreHelper = face_restoration_helper.FaceRestoreHelper(
                        upscale_factor=1, face_size=256)

    for i, filename in enumerate(pathnames):
        if i < args.start_from or i >= args.end_until:
            continue
        if i%100 == 0:
            print('processing {}th image'.format(i))
        # filename = args.content
        # print('Processing %s' % filename)
        basename = os.path.basename(filename).split('.')[0]
        scale = 1
        kernel_1d = np.array([[0.125],[0.375],[0.375],[0.125]])
        # print('Processing ' + os.path.basename(filename) + ' with vtoonify_' + args.backbone[0])
        
        try:
            if args.video:
                raise
            else:
                os.makedirs(args.output_path, exist_ok=True)
                # cropname = os.path.join(args.output_path, basename + '_input.jpg')
                savename = os.path.join(args.output_path, basename + '_vtoonify_' +  args.backbone[0] + '.png')
                # import pdb; pdb.set_trace()
                # for sake of rerun due to face detection failure, we first check if the output file exists when re-run
                if os.path.exists(savename):
                    continue

                frame = cv2.imread(filename)
                if frame.shape[0] == 512:
                    # frame = cv2.resize(frame, (1024,1024))
                    input_res = 512
                else:
                    input_res = 1024
                # import pdb; pdb.set_trace()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # We detect the face in the image, and resize the image so that the eye distance is 64 pixels.
                # Centered on the eyes, we crop the image to almost 400x400 (based on args.padding).
                if args.scale_image:
                    landmarks = align_net.get_landmarks(frame) # (98,2)
                    landmarks = landmark_98_to_68(landmarks) # (68,2)
                    
                    paras = get_crop_parameter_given_landmarks(landmarks, args.padding)
                    if paras is not None:
                        h,w,top,bottom,left,right,scale = paras
                        H, W = int(bottom-top), int(right-left)
                        # for HR image, we apply gaussian blur to it to avoid over-sharp stylization results
                        if scale <= 0.75:
                            frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
                        if scale <= 0.375:
                            frame = cv2.sepFilter2D(frame, -1, kernel_1d, kernel_1d)
                        frame = cv2.resize(frame, (w, h))[top:bottom, left:right]

                with torch.no_grad():
                    ### align face with facexlib
                    FaceRestoreHelper.clean_all()
                    FaceRestoreHelper.read_image(frame)
                
                    FaceRestoreHelper.get_face_landmarks_5()
                    # I = FaceRestoreHelper.align_warp_face()[..., ::-1]/ 255 # (256, 256, 3)
                    I = FaceRestoreHelper.align_warp_face()/255
                   
                    # print('shape after align_face():', np.array(I).shape)
                    
                    I = transform(I).unsqueeze(dim=0).to(device)
                    I = I.float() # float64 -> float32
                    
                    s_w = pspencoder(I)
                    s_w = vtoonify.zplus2wplus(s_w)
                    if vtoonify.backbone == 'dualstylegan':
                        if args.color_transfer:
                            s_w = exstyle
                        else:
                            s_w[:,:7] = exstyle[:,:7]
                    # NOTE: to make sure it can have aligned 1024
                    # x = transform(frame).unsqueeze(dim=0).to(device) 
                    x = I
                    # parsing network works best on 512x512 images, so we predict parsing maps on upsmapled frames
                    # followed by downsampling the parsing maps
                    x_p = F.interpolate(parsingpredictor(2*(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)))[0], 
                                        scale_factor=0.5, recompute_scale_factor=False).detach()
                    # we give parsing maps lower weight (1/16)
                    inputs = torch.cat((x, x_p/16.), dim=1)
                   
                    # d_s has no effect when backbone is toonify
                    y_tilde = vtoonify(inputs, s_w.repeat(inputs.size(0), 1, 1), d_s = args.style_degree)   
                    y_tilde = torch.clamp(y_tilde, -1, 1)

                # cv2.imwrite(cropname, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                # import pdb; pdb.set_trace()
                if input_res == 512:
                    y_tilde = F.interpolate(y_tilde, scale_factor=0.5, mode='bilinear', align_corners=False)
                save_image(y_tilde[0].cpu(), savename)
        except:
            print(colored('Error in processing {}'.format(filename), 'red'))

    print('Transfer style successfully!')