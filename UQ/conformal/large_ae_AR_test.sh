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
#SBATCH -o large_ae_AR.out

##### These are shell commands
date
cd /g/g14/katona1

echo 'activating'
. python.sh
cd mphys-surrogate-model

echo 'testing vanilla cp'
python3 UQ/conformal/cp_test.py erf_data/congestus/noadv_coal_200m_14400 -a AR -m full -s decoder
python3 UQ/conformal/cp_test.py erf_data/congestus/noadv_coal_200m_14400 -a AR -m full -s latent
python3 UQ/conformal/cp_test.py erf_data/congestus/noadv_coal_200m_14400 -a AR -m full -s full
python3 UQ/conformal/cp_test.py erf_data/congestus/noadv_coal_200m_14400 -a AR -m full -s mass
p=20 # validation split percent
echo "testing split cp, train-validation-test split: $((80 - p))-${p}-20"
python3 UQ/conformal/cp_test.py erf_data/congestus/noadv_coal_200m_14400 -a AR -m "split${p}" -s decoder
python3 UQ/conformal/cp_test.py erf_data/congestus/noadv_coal_200m_14400 -a AR -m "split${p}" -s latent
python3 UQ/conformal/cp_test.py erf_data/congestus/noadv_coal_200m_14400 -a AR -m "split${p}" -s full
python3 UQ/conformal/cp_test.py erf_data/congestus/noadv_coal_200m_14400 -a AR -m "split${p}" -s mass
k=20 # number of cross-validation folds
echo "testing cv+ with ${k} folds"
python3 UQ/conformal/cp_test.py erf_data/congestus/noadv_coal_200m_14400 -a AR -m "cv+${k}" -s decoder
python3 UQ/conformal/cp_test.py erf_data/congestus/noadv_coal_200m_14400 -a AR -m "cv+${k}" -s latent
python3 UQ/conformal/cp_test.py erf_data/congestus/noadv_coal_200m_14400 -a AR -m "cv+${k}" -s full
python3 UQ/conformal/cp_test.py erf_data/congestus/noadv_coal_200m_14400 -a AR -m "cv+${k}" -s mass