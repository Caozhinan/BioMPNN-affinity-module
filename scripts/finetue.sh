#!/bin/bash


# === 确保日志目录存在 ===
# mkdir -p ./Slurm

# === 加载 conda 环境 ===
source /xcfhome/ypxia/anaconda3/etc/profile.d/conda.sh
conda activate /xcfhome/ypxia/anaconda3/envs/proteinflow

# === 打印任务信息 ===

echo "Start time: $(date)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
# nvidia-smi



# === 运行训练脚本 ===
python /xcfhome/zncao02/BioMPNN_affinity/finetune.py \
    --data_csv /xcfhome/zncao02/dataset_bap/PDBBind/pdbbind_train.csv/ \
    --pretrained_ckpt /xcfhome/zncao02/BioMPNN_affinity/ckpt/best_model.pt \
    --mode train \
    --log_file finetune_training.log

# === 打印结束信息 ===
echo "End time: $(date)"