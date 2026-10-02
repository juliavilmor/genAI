#!/bin/bash
#SBATCH --job-name=test_genAI
#SBATCH --account=bsc72
#SBATCH --chdir=.
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --qos=acc_bscls

module purge

ml bsc/1.0
source activate genAI

#srun python -u tests/test.py --weights_file weights/weights_dm256_nh4_ff1024_nl6_downs50_new_epoch_50.pth --outdir tests/outdir --outname results
srun python -u tests/test.py --weights_file /gpfs/projects/cns120/weights/weights_dm256_nh4_ff2560_nl19_params_30M_epoch_50.pth --outdir tests/params_30M_new --outname results_epoch50
