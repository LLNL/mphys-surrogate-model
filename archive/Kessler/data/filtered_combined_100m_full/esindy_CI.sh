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
   data/filtered_combined_100m_full_processed.npy \
   -B 3000 \
   -t 39 321 1645 521 450 16 800 1000 1200 231 441 1220 1646 2917
echo 'done'


