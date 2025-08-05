#!/bin/bash

##### These lines are for Slurm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=110
#SBATCH --mem=50G
#SBATCH --exclusive
#SBATCH -J bin_psd
#SBATCH -t 0:10:00
#SBATCH -p pdebug
#SBATCH --mail-type=ALL
#SBATCH -A ml-uphys
#SBATCH -o output_%J.out

##### These are shell commands
date
cd /g/g14/katona1

echo 'activating'
. python.sh

echo 'running E-SINDy bootstrap'
cd Kessler
python3 esindy_parallel.py data/rico_test_processed.npy -t 5e-5 -f 0.02
echo 'done gathering coefficient samples'

echo 'data/rico_test_esindy.pkl' | python3 Kessler_hist_parallel.py
echo 'done plotting coefficient distributions'
