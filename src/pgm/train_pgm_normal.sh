#!/bin/bash

exp_name="sup_pgm_mimic"
# parents='a_r_s'
parents='a_r_s_f' 
mkdir -p "../../checkpoints/$parents/$exp_name"

python train_pgm.py \
    --data_dir='/vol/biodata/data/chest_xray/mimic-cxr-jpg-224/data/' \
    --csv_dir='/vol/biomedic3/tx1215/chexpert-dscm/src/mimic_meta' \
    --use_dataset='mimic' \
    --hps mimic192 \
    --setup='sup_pgm' \
    --exp_name=$exp_name \
    --parents_x age race sex finding\
    --lr=0.001 \
    --bs=32 \
    --wd=0.05 \
    --x_like='diag_dgauss' \
    --z_max_res=96 \
    --eval_freq=1 \
