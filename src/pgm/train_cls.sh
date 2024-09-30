#!/bin/bash
loss_norm="l1"
enc_net="resnet18"

exp_name="mimic_classifier_${enc_net}_${loss_norm}_lr3"

parents='a_r_s_f' 
mkdir -p "../../checkpoints/$parents/$exp_name"

python train_pgm.py \
    --data_dir='/vol/biodata/data/chest_xray/mimic-cxr-jpg-224/data/' \
    --csv_dir='/vol/biomedic3/tx1215/chexpert-dscm/src/mimic_meta' \
    --use_dataset='mimic' \
    --hps mimic224 \
    --setup='sup_determ' \
    --exp_name=$exp_name \
    --input_res=224 \
    --parents_x age race sex finding\
    --lr=1e-3 \
    --bs=32 \
    --wd=0.05 \
    --x_like='diag_dgauss' \
    --z_max_res=96 \
    --eval_freq=1 \
    --enc_net=$enc_net \
    --loss_norm=$loss_norm \
