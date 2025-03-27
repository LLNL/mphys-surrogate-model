#!/bin/bash

##### These lines are for Slurm
#SBATCH --mem=4G
#SBATCH -n 1
#SBATCH -J hp_eval
#SBATCH -t 00:30:00
#SBATCH -p pdebug
#SBATCH --mail-type=ALL
#SBATCH -A ml-uphys
#SBATCH -o myjob.%J.out

date
echo 'activating'

source ~/.bashrc
conda activate learning
export PYTHON=~/.conda/envs/learning/bin/python

echo 'running'
srun -n1 ${PYTHON} -u training_scripts/hyperparam_tuning.py

echo 'Done'

