#!/bin/bash

##### These lines are for Slurm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=110
#SBATCH --mem=20G
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
cd Kessler

echo 'running'
python3 Kessler_plot_cp_all_parallel.py data/box64 -m full
python3 Kessler_plot_cp_all_parallel.py data/box64 -m split
python3 Kessler_plot_cp_all_parallel.py data/box64 -m jackknife
python3 Kessler_plot_cp_all_parallel.py data/box64 -m cv+
echo 'done'