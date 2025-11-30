#!/bin/bash


# === 确保日志目录存在 ===
# mkdir -p ./Slurm

# === 加载 conda 环境 ===
source /xcfhome/ypxia/anaconda3/etc/profile.d/conda.sh
conda activate /xcfhome/ypxia/anaconda3/envs/proteinflow

# === 打印任务信息 ===

echo "Start time: $(date)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi



# === 运行训练脚本 ===
python /xcfhome/zncao02/BioMPNN_affinity/finetune.py \
    --data_csv /xcfhome/zncao02/dataset_bap/PDBBind/pdbbind_train.csv \
    --pretrained_ckpt /xcfhome/zncao02/BioMPNN_affinity/ckpt/best_model.pt \
    --mode train \
    --learning_rate 1e-5 \
    --weight_decay 1e-4 \
    --early_stop_epoch 15 \
    --epochs 100 \
    --log_file finetune_training2.log \
    --save_path finetuned_model_robust.pt


#     --data_csv /xcfhome/zncao02/dataset_bap/PDBBind/pdbbind_train.csv \
#     --mode train \
#     --graph_type Graph_EHIGN_5edges \
#     --batch_size 64 \
#     --epochs 200 \
#     --save_path best_model.pt

# === 打印结束信息 ===
echo "End time: $(date)"