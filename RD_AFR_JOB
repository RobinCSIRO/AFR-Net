#!/bin/bash
#SBATCH --job-name=RD_AFR
#SBATCH --time=156:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16g

# Application specific commands:
module load tensorflow/1.6.0-py36-gpu
module load python/3.6.1
module load opencv/3.2.0
cd /scratch1/wan246/DRainDrop/AFR_Net/RD_AFR
python main.py
