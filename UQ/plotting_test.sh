#!/bin/bash

##### These lines are for Slurm
#SBATCH --nodes=1
#SBATCH --ntasks=110
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
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

# Read into array
read -a arr <<< "$(cat conformal/test_idx.txt)"
total_items=${#arr[@]}

# Tasks per CPU core
tasks_per_cpu=$(( (total_items + SLURM_NTASKS - 1) / SLURM_NTASKS ))

# Launch one srun per CPU core
for (( core=0; core<$SLURM_NTASKS; core++ )); do
    start_index=$(( core * tasks_per_cpu ))
    end_index=$(( start_index + tasks_per_cpu ))
    if (( end_index > total_items )); then
        end_index=$total_items
    fi

    chunk=("${arr[@]:$start_index:$((end_index - start_index))}")

    if (( ${#chunk[@]} > 0 )); then
        echo "CPU core $core is testing elements: ${chunk[*]}"

        srun --exclusive -N1 -n1 python3 plotting.py congestus_coal_200m_9600 \
            -a SINDy \
            -t '0 30 60' \
            -m full \
            -s full \
            -g "${chunk[@]}" &
    fi
done

wait