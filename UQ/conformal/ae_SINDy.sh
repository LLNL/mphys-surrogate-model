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

export PYTHONUNBUFFERED=1

date
cd /g/g14/katona1 # changes this to home directory
echo 'activating'
. python.sh
cd mphys-surrogate-model

# --- Parameters ---
P_VALUES=(30 40)
CONGESTUS_ADD=40                         # constant addition for congestus
CONGESTUS_MAX=80                         
CONGESTUS_MIN=1
NUM_JOBS=${#P_VALUES[@]}                     # number of background jobs
NJOBS_PER_P=$((SLURM_CPUS_PER_TASK / NUM_JOBS))  # threads per Python process

echo "Total CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Parallel jobs: $NUM_JOBS, threads per job: $NJOBS_PER_P"

# --- Create log folder ---
mkdir -p logs
mkdir -p UQ/conformal/results/ae_SINDy

# --- Loop over p values in background ---
for p in "${P_VALUES[@]}"; do
    (
    echo "RICO split: ${p}-$((100 - p))"
    python3 UQ/conformal/ae_SINDy.py erf_data/RICO/noadv_coal_200m \
      -p "$p" -a 0.1 0.05 0.025 0.01 -j "$NJOBS_PER_P" \
      > "logs/RICO_p${p}_SINDy.log" 2>&1

    echo "congestus9600 split: ${p}-$((100 - p))"
    python3 UQ/conformal/ae_SINDy.py erf_data/congestus/noadv_coal_200m_9600 \
      -p "$p" -a 0.1 0.05 0.025 0.01 -j "$NJOBS_PER_P" \
      > "logs/congestus9600_p${p}_SINDy.log" 2>&1

    # congestus p = p + constant, then clamp
    p_small=$(( p + CONGESTUS_ADD ))
    (( p_small > CONGESTUS_MAX )) && p_small=$CONGESTUS_MAX
    (( p_small < CONGESTUS_MIN )) && p_small=$CONGESTUS_MIN

    echo "congestus split (p + ${CONGESTUS_ADD}): ${p_small}-$((100 - p_small)) (from base p=${p})"
    python3 UQ/conformal/ae_SINDy.py congestus_coal_200m_test \
      -p "$p_small" -a 0.1 0.05 0.025 0.01 -j "$NJOBS_PER_P" \
      > "logs/congestus_p${p_small}_SINDy.log" 2>&1

    echo "Finished pair: RICO p=${p} / congestus p=${p_small}"
    ) &
done

# Wait for all background jobs to finish
wait
echo "All jobs completed."
date