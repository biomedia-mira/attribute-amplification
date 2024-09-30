#!/bin/bash
exp_name='mimic_soft_label_not_age_and_not_disease_when_do_age_new_classifier_dscm_lr_1e4_lagrange_lr_1_damping_10'
# exp_name='mimic_soft_label_not_soft_race_dscm_lr_1e5_lagrange_lr_1_damping_10'
# Set parents as race, sex, age for now
parents='a_r_s_f' 
# parents='m_b_v_s'
mkdir -p "../../checkpoints/$parents/$exp_name"

sbatch <<EOT
#!/bin/bash

#SBATCH -p gpus                        # Partition (queue)
#SBATCH --nodes=1                      # Number of compute nodes
#SBATCH --ntasks-per-node=1            # Number of tasks per node (here 1 per GPU)
#SBATCH --gres=gpu:teslap40:1           # Number of GPUs per node, e.g. gpu:teslap40:2. Note: should match ntasks-per-node
#SBATCH --cpus-per-task=4              # Number of cpu cores per task
#SBATCH --mem=128gb                     # Memory pool for all cores
#SBATCH --output=../../checkpoints/$parents/$exp_name/slurm.%j.log   # Output and error log

nvidia-smi

# source conda environment
source activate tian_torch


srun python train_cf_soft_labelling.py \
    --data_dir='/vol/biodata/data/chest_xray/mimic-cxr-jpg-224/data/' \
    --csv_dir='/vol/biomedic3/tx1215/chexpert-dscm/src/mimic_meta' \
    --use_dataset='mimic' \
    --hps mimic192 \
    --exp_name=$exp_name \
    --parents_x age race sex finding\
    --lr=1e-4 \
    --lr_lagrange=0.1 \
    --damping=10 \
    --bs=10 \
    --wd=0.05 \
    --x_like='diag_dgauss' \
    --z_max_res=96 \
    --eval_freq=1 \
    --predictor_path='../../checkpoints/a_r_s_f/Fabio_mimic_sup_aux_resnet18_wd0.1_drop0.2_saug_bs32/checkpoint.pt' \
    --pgm_path='../../checkpoints/a_r_s_f/sup_pgm_mimic/checkpoint.pt' \
    --vae_path='../../checkpoints/a_r_s_f/mimic_beta9_gelu_dgauss_1_lr3/checkpoint.pt'
EOT