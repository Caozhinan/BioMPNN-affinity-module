#!/bin/bash
source /xcfhome/ypxia/anaconda3/etc/profile.d/conda.sh
conda activate /xcfhome/ypxia/anaconda3/envs/proteinflow

##测试
# python /xcfhome/zncao02/BioMPNN_affinity/run_pipline.py \
#     --data_csv /xcfhome/zncao02/BioMPNN_affinity/scripts/core_set.csv \
#     --mode test \
#     --skip_preprocess \
#     --skip_graph \
#     --save_path /xcfhome/zncao02/BioMPNN_affinity/finetuned_model.pt