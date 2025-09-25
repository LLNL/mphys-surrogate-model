#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=15                # one task per seed (listed 15)
#SBATCH --cpus-per-task=3          # 3 cores per python proc
#SBATCH --mem=120G
#SBATCH --exclusive
#SBATCH -J bin_psd
#SBATCH -t 1:00:00
#SBATCH -p pdebug
#SBATCH --mail-type=ALL
#SBATCH -A ml-uphys
#SBATCH -o output_%J.out

export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 NUMEXPR_NUM_THREADS=3

date
cd /g/g14/katona1
echo 'activating'
. python.sh
cd mphys-surrogate-model

echo 'plotting RICO cases'
python3 UQ/plot.py erf_data/RICO/noadv_coal_200m -t "0 100" -s full -g 1452 -p 40 -title n
python3 UQ/plot.py erf_data/RICO/noadv_coal_200m -t "0 100" -s full -g 1701 -p 40 -title n
# srun --cpu-bind=cores -l bash -lc '
#   seeds=(1452 2681 323 938 2571 1647 1102 939 2724 2115 1701 728 523 3176 2652)

#   idx=${SLURM_PROCID:?}
#   if [[ $idx -ge ${#seeds[@]} ]]; then
#     echo "Task $idx: no seed (out of range), exiting."
#     exit 0
#   fi

#   seed=${seeds[$idx]}
#   echo "Task $idx -> seed $seed"
#   python3 UQ/plot.py erf_data/RICO/noadv_coal_200m -t "0 100" -s full -g "$seed" -p 40 -title n
# '

echo 'plotting congestus test cases'
python3 UQ/plot.py congestus_coal_200m_test -t '0 60' -s full -g 59 -p 80 -title n
python3 UQ/plot.py congestus_coal_200m_test -t '0 60' -s full -g 45 -p 80 -title n
echo 'plotting congestus 9600 cases'
python3 UQ/plot.py erf_data/congestus/noadv_coal_200m_9600 -t '0 60' -s full -g 1146 -p 40 -title n
echo 'done'

date