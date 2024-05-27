#!/bin/bash
#SBATCH -J esnli_lmeans_multinode_st3_third_batch
#SBATCH -o esnli_lmeans_multinode_st3_third_batch.o%j
#SBATCH -e esnli_lmeans_multinode_st3_third_batch.e%j
#SBATCH -p skx
#SBATCH -N 64
#SBATCH -n 64
#SBATCH -t 24:00:00
#SBATCH --mail-user=sam.su@utexas.edu
#SBATCH --mail-type=ALL
#SBATCH -A CCR24018

# Run the tasks
for i in {128..191}; do
  start=$((55 * i))
  end=$((start + 55))
  srun --nodes=1 --ntasks=1 --exclusive --job-name=task$i bash -c "
    source /work/06782/ysu707/ls6/conda/etc/profile.d/conda.sh
    conda activate wrapperbox
    cd /work/06782/ysu707/ls6/WrapperBox/MinimalSubsetToFlipPredictions
    python3 main.py --config config/esnli/lmeans_less/deberta_mean_with_attention_LMeans_${start}_${end}.json
  " &
done

wait
