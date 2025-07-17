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
python3 congestus_on_rico_plot_parallel.py -m full
python3 congestus_on_rico_plot_parallel.py -m split
python3 congestus_on_rico_plot_parallel.py -m jackknife
python3 congestus_on_rico_plot_parallel.py -m cv+
echo 'done'