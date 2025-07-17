#!/bin/bash

##### These lines are for Slurm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=110
#SBATCH --mem=50G
#SBATCH --exclusive
#SBATCH -J bin_psd
#SBATCH -t 0:10:00
#SBATCH -p pbatch
#SBATCH --mail-type=ALL
#SBATCH -A ml-uphys
#SBATCH -o output_%J.out

##### These are shell commands
date
cd /g/g14/katona1

echo 'activating'
. python.sh

echo 'Kessler/data/rico_test.nc' | python3 xyz_dump_filtered.py
echo 'done extracting coordinates'

cd Kessler/data/
echo 'running'
python3 check_cor_parallel.py rico_test
echo 'done computing correlations'

python3 analyze_cor.py rico_test
echo 'done eturning correlation results'