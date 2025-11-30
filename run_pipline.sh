#!/bin/bash
source /xcfhome/ypxia/anaconda3/etc/profile.d/conda.sh
conda activate /xcfhome/ypxia/anaconda3/envs/proteinflow

##测试
python /xcfhome/zncao02/BioMPNN_affinity/run_pipline.py \
    --data_csv /xcfhome/zncao02/BioMPNN_affinity/scripts/core_set.csv \
    --mode test \
    --skip_preprocess \
    --skip_graph \
    --save_path  /xcfhome/zncao02/BioMPNN_affinity/ckpt/BN_pre.pt \
    --log_file /xcfhome/zncao02/BioMPNN_affinity/log/BN_pre_test.log \
    --output_csv /xcfhome/zncao02/BioMPNN_affinity/results/BN_pre_test.csv

##训练
# python /xcfhome/zncao02/BioMPNN_affinity/run_pipline.py \
#     --data_csv /xcfhome/zncao02/dataset_bap/PDBBind/pdbbind_train.csv \
#     --mode train \
#     --skip_preprocess \
#     --skip_graph \
#     # --save_path /xcfhome/zncao02/BioMPNN_affinity/pdbbind_pretrained.pt \
    