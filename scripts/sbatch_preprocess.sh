#!/bin/bash
#SBATCH -J pyg—extract
#SBATCH -p helium  
#SBATCH -N 1 -n 1
#SBATCH --qos=qcpu
#SBATCH --cpus-per-task=16        # 为这个任务分配 16 个 CPU 核


source /xcfhome/ypxia/anaconda3/etc/profile.d/conda.sh
ca /xcfhome/ypxia/anaconda3/envs/proteinflow
python batch_custom_preprocess.py \
    --csv /xcfhome/zncao02/dataset_bap/test_set/pdbbind/csv/core_set.csv \
    --distance 5.0 \
    --n_jobs 4 \
    --batch_size 1000 \
    --timeout 300