#!/bin/bash
#SBATCH -J esnli_lmeans_multinode
#SBATCH -o esnli_lmeans_multinode.o%j
#SBATCH -e esnli_lmeans_multinode.e%j
#SBATCH -p normal
#SBATCH -N 50
#SBATCH -n 50
#SBATCH -t 48:00:00
#SBATCH --mail-user=sam.su@utexas.edu
#SBATCH --mail-type=ALL
#SBATCH -A CCR24018

# Run the tasks
for i in {0..49}; do
  start=$((200 * i))
  end=$((start + 200))
  srun --nodes=1 --ntasks=1 --exclusive --job-name=task$i bash -c "
    source /work/06782/ysu707/ls6/conda/etc/profile.d/conda.sh
    conda activate wrapperbox
    cd /work/06782/ysu707/ls6/WrapperBox/MinimalSubsetToFlipPredictions
    python3 main.py --config config/esnli/lmeans/deberta_mean_with_attention_LMeans_${start}_${end}.json
  " &
done

wait
