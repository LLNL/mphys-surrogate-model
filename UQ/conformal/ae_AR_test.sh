#!/bin/bash
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

set -euo pipefail
export PYTHONUNBUFFERED=1

date
cd /g/g14/katona1
. python.sh
cd mphys-surrogate-model
mkdir -p logs

DATASETS=( "erf_data/RICO/noadv_coal_200m" "congestus_coal_200m_test" "erf_data/congestus/noadv_coal_200m_9600")
P_VALUES=(30 40)
CONGESTUS_ADD=40                         # constant addition for congestus
CONGESTUS_MAX=80                         
CONGESTUS_MIN=1
S_VALUES=( decoder latent full mass )

THREADS_PER_STEP=3
for dataset in "${DATASETS[@]}"; do
  for p in "${P_VALUES[@]}"; do
    # pick effective p for this dataset
    if [[ "$(basename "$dataset")" == "congestus_coal_200m_test" ]]; then
      p_eff=$(( p + CONGESTUS_ADD ))
      (( p_eff > CONGESTUS_MAX )) && p_eff=$CONGESTUS_MAX
      (( p_eff < CONGESTUS_MIN )) && p_eff=$CONGESTUS_MIN    
    else
      p_eff=$p
    fi
    p_tag="p${p_eff}"

    for s in "${S_VALUES[@]}"; do
      tag="$(basename "$dataset")_${p_tag}_${s}_AR"
      echo "Launching: dataset=$(basename "$dataset"), p=${p_eff}, s=${s}"

      srun --exclusive -c "$THREADS_PER_STEP" --cpu-bind=cores --hint=nomultithread -J "$tag" bash -lc "
        export OMP_NUM_THREADS=$THREADS_PER_STEP
        export MKL_NUM_THREADS=$THREADS_PER_STEP
        export OPENBLAS_NUM_THREADS=$THREADS_PER_STEP
        export NUMEXPR_NUM_THREADS=$THREADS_PER_STEP
        export CPU_THREADS_PER_STEP=$THREADS_PER_STEP
        python3 -u UQ/conformal/cp_test.py '$dataset' -a AR -p $p_eff -s $s
      " > "logs/${tag}.log" 2>&1 &
    done
  done
done
wait

echo "All jobs completed."
date