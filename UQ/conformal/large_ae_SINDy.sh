#!/bin/bash

##### These lines are for Slurm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=110
#SBATCH --mem=120G
#SBATCH --exclusive
#SBATCH -J bin_psd
#SBATCH -t 12:00:00
#SBATCH -p pbatch
#SBATCH --mail-type=ALL
#SBATCH -A ml-uphys
#SBATCH -o output_%J.out

##### These are shell commands
date
cd /g/g14/katona1

echo 'activating'
. python.sh
cd mphys-surrogate-model

echo 'starting vanilla cp'
python3 UQ/conformal/ae_SINDy.py erf_data/congestus/noadv_coal_200m_14400 -m full -e 200 -b 200 -a 0.1 0.05 0.025 0.01
p=20 # validation split percent
echo "starting split cp, train-validation-test split: $((80 - p))-${p}-20"
python3 UQ/conformal/ae_SINDy.py erf_data/congestus/noadv_coal_200m_14400 -m "split${p}" -e 200 -b 200 -a 0.1 0.05 0.025 0.01