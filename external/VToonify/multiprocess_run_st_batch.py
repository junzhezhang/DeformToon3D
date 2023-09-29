# run this script using python. cannot invoke srun in srun
import tqdm; from multiprocessing import Pool
import os
num_gpus = 16
total_runs = 100
def process_one(process_id):
    # pass
    print('process one ...', process_id)
    os.system("srun -u --partition=Zoetrope -n1 --gres=gpu:1 --ntasks-per-node=1 --job-name mp_style python style_transfer_batch.py --scale_image --content /mnt/cache/zhangjunzhe/data/FFHQ/generated_images1024x1024/v9_50k")
def main():
    with Pool(num_gpus) as p:
        r = list(p.imap(process_one, range(total_runs)))

if __name__ == '__main__':
    main()