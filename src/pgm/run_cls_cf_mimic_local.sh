#!/bin/bash

use_data="cf"
eval_data="real_cf"

exp_name="local_mimic_with_CF_train_${use_data}_valid_${eval_data}_finding_classifier_resnet18_CrossEntro_lr3_slurm"
parents='a_r_s_f'
dscm_dir='mimic_soft_label_dscm_lr_1e5_lagrange_lr_1_damping_10'
which_checkpoint='15500_checkpoint'

mkdir -p "../checkpoints/$parents/${exp_name}"

source activate tian_torch
python train_cls_cf_mimic.py \
    --use_data=${use_data} \
    --eval_data=${eval_data} \
    --use_dataset="mimic_cfs" \
    --lr=1e-3 \
    --bs=32 \
    --wd=0.05 \
    --csv_dir="/vol/biomedic3/tx1215/chexpert-dscm/CF_DATA/${dscm_dir}/${which_checkpoint}" \
    --parents_x age race sex finding \
    --exp_name=${exp_name} \
    --cls_net='resnet18' \
    --patience_for_scheduler=10 \
    --metric_to_monitor_mode='min' \
    --epochs=1000 \
    --input_res=224 \ 
    --eval_freq=1 \          
