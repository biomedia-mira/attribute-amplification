#!/bin/bash

use_data="cf"
eval_data="real"
parents='a_r_s_f'

### 
# which_cf='sex'
which_cf='race'
# which_cf='finding'

setup='sup_determ'

### For soft label CFs
dscm_dir="mimic_soft_label_not_age_and_not_disease_when_do_age_new_classifier_dscm_lr_1e4_lagrange_lr_1_damping_10"
which_checkpoint="45000_checkpoint"

### For hard label CFs
# dscm_dir="mimic_dscm_new_classifier_lr_1e4_lagrange_lr_1_damping_10"
# which_checkpoint="12000_checkpoint"

### For non CF training CFs
# dscm_dir="mimic_dscm_new_classifier_lr_1e4_lagrange_lr_1_damping_10"
# which_checkpoint="0_checkpoint"

exp_name="mimic_train_${use_data}_${which_cf}_val_${eval_data}_classifier_resnet18_lr4_slurm_${setup}"
mkdir -p "checkpoints/${dscm_dir}/${which_checkpoint}/$parents/$exp_name"

# source conda environment
source activate tian_torch
python train_cls_cf_mimic.py \
    --use_data=$use_data \
    --eval_data="${eval_data}" \
    --which_cf=$which_cf \
    --dscm_dir="${dscm_dir}" \
    --setup="${setup}" \
    --which_checkpoint="${which_checkpoint}" \
    --use_dataset="mimic_cfs" \
    --loss_norm="l2" \
    --lr=1e-4 \
    --bs=32 \
    --wd=0.1 \
    --csv_dir="/vol/biomedic3/tx1215/chexpert-dscm/CF_DATA/${dscm_dir}/${which_checkpoint}" \
    --parents_x age race sex finding \
    --exp_name=${exp_name} \
    --enc_net='resnet18' \
    --epochs=1000 \
    --input_res=224 \ 
    --eval_freq=1 \    
    --x_like='diag_dgauss' \
    --z_max_res=96 \
    --eval_freq=1 