#!/bin/bash
#SBATCH -J esnli_yang_fast_multinode
#SBATCH -o esnli_yang_fast_multinode.o%j
#SBATCH -e esnli_yang_fast_multinode.e%j
#SBATCH -p skx
#SBATCH -N 29
#SBATCH -n 29
#SBATCH -t 24:00:00
#SBATCH --mail-user=sam.su@utexas.edu
#SBATCH --mail-type=ALL
#SBATCH -A CCR24018

# Run the tasks
for i in {0..28}; do
  start=$((350 * i))  # Calculate start as 450 times i
  end=$((start + 350)) # Calculate end as start + 450
  srun --nodes=1 --ntasks=1 --exclusive --job-name=task$i bash -c "
    source /work/06782/ysu707/ls6/conda/etc/profile.d/conda.sh
    conda activate wrapperbox
    cd /work/06782/ysu707/ls6/WrapperBox/MinimalSubsetToFlipPredictions
    python3 main.py --config config/esnli/yang_fast/deberta_mean_with_attention_yang_fast_${start}_${end}.json
  " &
done

wait
