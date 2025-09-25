#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1                 # we'll launch 3 separate steps ourselves
#SBATCH --cpus-per-task=9          # not used by per-step sruns; harmless
#SBATCH --mem=90G                  # shared across the node
#SBATCH -J bin_psd
#SBATCH -t 1:00:00
#SBATCH -p pdebug
#SBATCH --mail-type=ALL
#SBATCH -A ml-uphys
#SBATCH -o output_%J.out

# Keep numerical libs from oversubscribing threads
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3

date
cd /g/g14/katona1
echo 'activating'
. python.sh
cd mphys-surrogate-model

# Launch 3 independent steps (each gets 3 CPUs); failures won't nuke the others
srun --exclusive -N1 -n1 -c3 --cpu-bind=cores bash -lc \
  'python3 UQ/errors.py erf_data/RICO/noadv_coal_200m -s all -p 40' &

srun --exclusive -N1 -n1 -c3 --cpu-bind=cores bash -lc \
  'python3 UQ/errors.py congestus_coal_200m_test -s all -p 80' &

srun --exclusive -N1 -n1 -c3 --cpu-bind=cores bash -lc \
  'python3 UQ/errors.py erf_data/congestus/noadv_coal_200m_9600 -s all -p 40' &

wait
echo 'done'
date