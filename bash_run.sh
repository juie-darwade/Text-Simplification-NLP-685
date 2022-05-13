#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=40G  # Requested Memory
#SBATCH -p gpu-long  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 48:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID

module load cuda/10
/modules/apps/cuda/10.1.243/samples/bin/x86_64/linux/release/deviceQuery
/opt/conda/bin/conda
python -u "/home/jupyter/Project/run.py" train --base_path "/home/jupyter/Project/" --src_train "dataset/src_train.txt" --src_valid "dataset/src_valid.txt" --tgt_train "dataset/tgt_train.txt" --tgt_valid "dataset/tgt_valid.txt" --checkpoint_path "checkpoint/model_ckpt.pt" --seed 540