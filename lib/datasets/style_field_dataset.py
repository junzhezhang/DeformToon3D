import torch.utils.data as data
import torch
import numpy as np
import imageio 
import glob
import os
from PIL import Image
import torchvision

class StyleFieldDataset(data.Dataset):
    def __init__(self, data_dir=None, source_data=None, style_data=None, train_split_ratio=1):
        self.meta_path = os.path.join(data_dir, f'latents.pth')
        self.source_img_path = os.path.join(data_dir, source_data)
        self.data_dir = data_dir
        self.meta_dict = torch.load(self.meta_path)
        if '+' not in style_data:
            self.n_styles = 1
            style_img_path = os.path.join(data_dir, style_data)
            style_pathnames = sorted(glob.glob(style_img_path + '/*'))
            
            # NOTE: the order of img_stems is tally with meta_dict lists
            all_img_stems = [os.path.basename(p).split('_')[0] for p in style_pathnames]
            if train_split_ratio < 1:
                img_stems = all_img_stems[:int(len(all_img_stems) * train_split_ratio)]
            else:
                img_stems = all_img_stems
            self.imgs_stem_pairs = [(0, style_data,stem) for stem in img_stems]
        else:
            # The train set contains multiple style data, concatentated with '+'
            style_datas = style_data.split('+')
            self.n_styles = len(style_datas)
            self.imgs_stem_pairs = []
            if train_split_ratio < 1:
                raise "train_split_ratio < 1 is not implemented for multiple style data!!!"
            for style_id, this_style_data in enumerate(style_datas):
                style_pathnames = sorted(glob.glob(os.path.join(data_dir, this_style_data, '*')))
                all_img_stems = [os.path.basename(p).split('_')[0] for p in style_pathnames]
                self.imgs_stem_pairs += [(style_id, this_style_data, stem) for stem in all_img_stems]
                print('Style {} {} with {} images'.format(style_id, this_style_data, len(all_img_stems)))

        # dict_keys(['z', 'cam_extrinsics', 'focals', 'near', 'far', 'locations'])
        
        print('Dataset with {} images'.format(len(self.imgs_stem_pairs)))
    
    def __getitem__(self, index):
        try:
            style_id, style_data, stem = self.imgs_stem_pairs[index]
            cartoon_pathname = glob.glob(os.path.join(self.data_dir, style_data, stem + '*'))[0]
            cartoon_img = Image.open(cartoon_pathname)
            real_img = Image.open(os.path.join(self.source_img_path, stem + '_fullhd.png'))
            cartoon_img = torchvision.transforms.functional.to_tensor(cartoon_img)
            real_img = torchvision.transforms.functional.to_tensor(real_img)
            if 'cam_extrinsics' in self.meta_dict[stem]:
                ret = {
                    'real_img': real_img,
                    'cartoon_img': cartoon_img,
                    'z': self.meta_dict[stem]['z'],
                    'cam_extrinsics': self.meta_dict[stem]['cam_extrinsics'],
                    'focal': self.meta_dict[stem]['focal'],
                    'near': self.meta_dict[stem]['near'],
                    'far': self.meta_dict[stem]['far'],
                    'location': self.meta_dict[stem]['location'],
                    'style_id': style_id,
                    'stem': int(stem),
                }
            elif 'camera_params' in self.meta_dict[stem]:
                ret = {
                    'real_img': real_img,
                    'cartoon_img': cartoon_img,
                    'z': self.meta_dict[stem]['z'],
                    'camera_params' : self.meta_dict[stem]['camera_params'],
                    'style_id': style_id,
                }
        except:
            import pdb; pdb.set_trace()

        return ret

    def __len__(self):
        return len(self.imgs_stem_pairs)

