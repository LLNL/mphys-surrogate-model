#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=3                 # one task per command
#SBATCH --cpus-per-task=3          # 3 CPUs (threads) per task
#SBATCH --mem=10G                  # shared across the node
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
cd mphys-surrogate-model/UQ/conformal

# ensure output directory exists for redirection
mkdir -p test_idx

srun --exclusive -N1 -n1 -c3 --cpu-bind=cores bash -lc \
  'python3 extract_test_idx.py erf_data/RICO/noadv_coal_200m -p 40 > test_idx/RICO_test_idx.txt' &

srun --exclusive -N1 -n1 -c3 --cpu-bind=cores bash -lc \
  'python3 extract_test_idx.py congestus_coal_200m_test -p 80 > test_idx/congestus_test_idx.txt' &

srun --exclusive -N1 -n1 -c3 --cpu-bind=cores bash -lc \
  'python3 extract_test_idx.py erf_data/congestus/noadv_coal_200m_9600 -p 40 > test_idx/congestus_9600_idx.txt' &
wait

echo 'done'
date