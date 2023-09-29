# DeformToon3D: Deformable 3D Toonification from Neural Radiance Fields (ICCV 2023)

#### Project page: [link](https://www.mmlab-ntu.com/project/deformtoon3d/) 
#### Paper: [link](https://arxiv.org/abs/2309.04410)

<div align="center">
<img src=./assets/teaser.png>
</div>

## Environment setup

```
conda env create --file environment.yml
```
Alternatively, you may refer to StyleSDF environment setup.

## Generate dataset
### Generate real-space images and latents with pre-trained StyleSDF
NOTE: to manually change output directories before running
NOTE: You can generate more data and save it as latents_eval.pth for visualization.
The pre-trained models can be downloaded by running python ```download_models.py.```
```
python generate_images_and_latents.py \
--style_field_option no_style_field --elastic_loss 0 --adaptive_style_mixing \
--output_root_dir data --identities 1000 
```

### Generate stylized data with pre-trained DualStyleGAN
Run the following script to generate 10 styliized data corresponding to the real-space data generated above.
NOTE: refer to DualStyle for more details.
```
bash generate_stylized_data.sh
```

## Training
The default style_data contains 10 styles and base style (real_space).
Train the 1st 50 epochs without GAN loss for sake of speed.
```
python main.py  \
--jobname job_name \
--n_epoch 50
```
Continue to train 50 epochs with GAN loss
```
python main.py  \
--style_batch 1 \
--jobname job_name \
--n_epoch 50 \
--gan_loss 0.05 \
--continue_training 49
```

## Quick Demo
You may run Generate real-space images and latents section to get the latents.pth or latents_eval.pth. Note our method generalizes well to unseen latents.
```
python main.py  \
--jobname job_name \
--exp_mode visualize_video --n_styles 11 --num_frames 250 \
--given_subject_list 1000-1010 --style_id 7 
```



## Acknowledgments
This code is built upon codebase of [StyleSDF](https://github.com/royorel/StyleSDF), and it also contains submodules including [DualStyleGAN](https://github.com/williamyang1991/DualStyleGAN), [VToonify](https://github.com/williamyang1991/VToonify), [PerceptualSimilarity](https://github.com/shubhtuls/PerceptualSimilarity), and [facexlib](https://github.com/xinntao/facexlib).


## Citation
```
@inproceedings{zhang2023deformtoon3d,
 title = {DeformToon3D: Deformable 3D Toonification from Neural Radiance Fields},
 author = {Junzhe Zhang, Yushi Lan, Shuai Yang, Fangzhou Hong, Quan Wang, Chai Kiat Yeo, Ziwei Liu, Chen Change Loy},
 booktitle = {ICCV},
 year = {2023}}
```
