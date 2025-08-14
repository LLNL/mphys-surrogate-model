#!/bin/bash

##### These lines are for Slurm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=110
#SBATCH --mem=120G
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
cd mphys-surrogate-model/UQ

python3 plotting.py congestus_coal_200m_9600 \
            -a SINDy \
            -t '0 30 60' \
            -m cv+20 \
            -s full \
            -g 404 121 249
python3 plotting.py congestus_coal_200m_9600 \
            -a SINDy \
            -t '0 30 60' \
            -m cv+20 \
            -s full \
            -g 387 35 477 65
python3 plotting.py congestus_coal_200m_9600 \
            -a SINDy \
            -t '0 30 60' \
            -m cv+20 \
            -s full \
            -g 121 235 135 7
python3 plotting.py congestus_coal_200m_9600 \
            -a SINDy \
            -t '0 30 60' \
            -m cv+20 \
            -s full \
            -g 11
python3 plotting.py congestus_coal_200m_9600 \
            -a SINDy \
            -t '0 30 60' \
            -m cv+20 \
            -s full \
            -g 123 370 282 289 293 51 78