#!/bin/bash
#SBATCH -J pyg—extract
#SBATCH -p helium  
#SBATCH -N 1 -n 1
#SBATCH --qos=qcpu
#SBATCH --cpus-per-task=16        # 为这个任务分配 16 个 CPU 核
#SBATCH --nodelist=f131
ca /xcfhome/ypxia/anaconda3/envs/proteinflow
python /xcfhome/zncao02/BioMPNN_affinity/custom_preprocess.py --csv /xcfhome/zncao02/dataset_bap/Bindingnetv2/csv/high_valid.csv --n_jobs 16