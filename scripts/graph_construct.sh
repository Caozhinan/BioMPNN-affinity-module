#!/bin/bash  
source /xcfhome/ypxia/anaconda3/etc/profile.d/conda.sh  
conda activate /xcfhome/ypxia/anaconda3/envs/proteinflow  
# export DGLBACKEND=pytorch
# export DGL_DOWNLOAD=0
# 清除PYTHONPATH,避免加载用户本地包  
# unset PYTHONPATH  
# export PYTHONNOUSERSITE=1
  
# 设置库路径  
export LD_LIBRARY_PATH=/xcfhome/ypxia/anaconda3/envs/proteinflow/lib:$LD_LIBRARY_PATH
  
# 直接使用conda环境的Python可执行文件  
/xcfhome/ypxia/anaconda3/envs/proteinflow/bin/python graph_constructor.py \
    --csv_file /xcfhome/zncao02/dataset_bap/PDBBind/pdbbind_train.csv \
    --graph_type Graph_EHIGN_5edges \
    --dis_threshold 5.0 \
    --num_process 10 \
    --create