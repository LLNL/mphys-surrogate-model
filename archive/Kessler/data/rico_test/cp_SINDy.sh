#!/bin/bash

##### These lines are for Slurm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=110
#SBATCH --mem=120G
#SBATCH --exclusive
#SBATCH -J bin_psd
#SBATCH -t 0:30:00
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
python3 cp_SINDy_parallel.py \
   data/rico_test_processed.npy \
   -m full \
   -a 0.25 0.1 0.05 0.01
python3 cp_SINDy_parallel.py \
   data/rico_test_processed.npy \
   -t 0.4 \
   -m split50 \
   -a 0.25 0.1 0.05 0.01
python3 cp_SINDy_parallel.py \
   data/rico_test_processed.npy \
   -m jackknife \
   -a 0.25 0.1 0.05 0.01
# python3 cp_SINDy_parallel.py \
#    data/rico_test_processed.npy \
#    -m jackknife+ \
#    -a 0.25 0.1 0.05 0.01
python3 cp_SINDy_parallel.py \
   data/rico_test_processed.npy \
   -a 0.25 0.1 0.05 0.01
echo 'done'