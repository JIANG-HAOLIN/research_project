#!/bin/bash -l

#Slurm parameters
#SBATCH --job-name=noi_lb
#SBATCH --output=4oC_conv_encoder_U_tm%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=6-23:00:00
#SBATCH --mem=64G
# SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --qos=batch
# SBATCH --gpus=geforce_rtx_2080_ti:1
# SBATCH --gpus=geforce_gtx_titan_x:1
# SBATCH --gpus=geforce_gtx_1080_ti:1
#SBATCH --gpus=rtx_a5000:1

# Activate everything you need
#echo $PYENV_ROOT
#echo $PATH
export PATH="/usrhomes/s1434/anaconda3/envs/myenv/bin:/usrhomes/s1434/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"
conda activate /usrhomes/s1434/anaconda3/envs/myenv
# Run your python code

python train_supervised.py --name oasis_cityscapes --dataset_mode cityscapes --gpu_ids 0 \
--dataroot /data/public/cityscapes --batch_size 4  \
--model_supervision 2 --netG 41322 --channels_G 64 --num_epochs 500 \
--checkpoints_dir ./checkpoints/checkpoints_b4_bipaDEC_catLabel_skipSPd_3Dnoise_noisylb_2