#!/bin/bash

##### These lines are for Slurm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --mem=50G
#SBATCH --exclusive
#SBATCH -J bin_psd
#SBATCH -t 1:00:00
#SBATCH -p pbatch
#SBATCH --mail-type=ALL
#SBATCH -A ml-uphys
#SBATCH -o output_%J.out

##### These are shell commands
date
cd /g/g14/katona1

echo 'activating'
. python.sh

cd Kessler
echo 'training SINDy model'
python3 sindy.py data/rico_test_processed.npy -t 5e-5
echo 'running SINDy plotting script'
echo 'data/rico_test_processed.npy' | python3 Kessler_plot_SINDy_all_parallel.py 
echo 'done'