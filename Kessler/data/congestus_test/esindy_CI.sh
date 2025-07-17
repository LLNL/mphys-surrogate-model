#!/bin/bash

##### These lines are for Slurm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=110
#SBATCH --mem=250G
#SBATCH --exclusive
#SBATCH -J bin_psd
#SBATCH -t 1:00:00
#SBATCH -p pdebug
#SBATCH --mail-type=ALL
#SBATCH -A ml-uphys
#SBATCH -o output_%J.out

##### These are shell commands
date
cd /g/g14/katona1

echo 'activating'
. python.sh
cd Kessler

echo 'running'
python3 Kessler_plot_E-SINDy_all_parallel.py \
   data/congestus_test_processed.npy \
   -B 1000 \
   -t 39 321 521 450 16 231 441 391 460 254
echo 'done'
