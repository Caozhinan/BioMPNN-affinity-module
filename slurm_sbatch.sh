#!/bin/bash
#SBATCH -J BioMPNN-affinity
#SBATCH -p boron,carbon,iron
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8 
#SBATCH -o ./Slurm/BioMPNN-affinity_%j.out
#SBATCH -e ./Slurm/BioMPNN-affinity_%j.out

# === 确保日志目录存在 ===
mkdir -p ./Slurm

# === 加载 conda 环境 ===
source /xcfhome/ypxia/anaconda3/etc/profile.d/conda.sh
conda activate /xcfhome/ypxia/anaconda3/envs/proteinflow

# === 打印任务信息 ===
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# === 进入工作目录 ===
cd /xcfhome/zncao02/BioMPNN_affinity

# === 运行训练脚本 ===
python train.py

# === 打印结束信息 ===
echo "End time: $(date)"