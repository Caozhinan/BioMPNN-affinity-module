source /xcfhome/ypxia/anaconda3/etc/profile.d/conda.sh
conda activate /xcfhome/ypxia/anaconda3/envs/proteinflow
export LD_LIBRARY_PATH=/xcfhome/ypxia/anaconda3/envs/proteinflow/lib:$LD_LIBRARY_PATH
/xcfhome/ypxia/anaconda3/envs/proteinflow/bin/python batch_custom_preprocess.py \
    --csv /xcfhome/zncao02/dataset_bap/PDBBind/pdbbind_train.csv \
    --distance 5.0 \
    --n_jobs 8 \
    --batch_size 1000 \
    --timeout 300