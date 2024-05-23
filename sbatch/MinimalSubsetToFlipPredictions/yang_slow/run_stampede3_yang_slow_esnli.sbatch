#!/bin/bash
#SBATCH -J esnli_yang_slow_multinode
#SBATCH -o esnli_yang_slow_multinode.o%j
#SBATCH -e esnli_yang_slow_multinode.e%j
#SBATCH -p skx
#SBATCH -N 22
#SBATCH -n 22
#SBATCH -t 48:00:00
#SBATCH --mail-user=sam.su@utexas.edu
#SBATCH --mail-type=ALL
#SBATCH -A CCR24018

# Run the tasks
for i in {0..21}; do
  start=$((450 * i))  # Calculate start as 450 times i
  end=$((start + 450)) # Calculate end as start + 450
  srun --nodes=1 --ntasks=1 --exclusive --job-name=task$i bash -c "
    source /work/06782/ysu707/ls6/conda/etc/profile.d/conda.sh
    conda activate wrapperbox
    cd /work/06782/ysu707/ls6/WrapperBox/MinimalSubsetToFlipPredictions
    python3 main.py --config config/esnli/yang_slow/deberta_mean_with_attention_yang_slow_${start}_${end}.json
  " &
done

wait
