#!/bin/bash
#SBATCH -J esnli_evaluate_yang_fast_multinode3
#SBATCH -o esnli_evaluate_yang_fast_multinode3.o%j
#SBATCH -e esnli_evaluate_yang_fast_multinode3.e%j
#SBATCH -p skx
#SBATCH -N 64
#SBATCH -n 64
#SBATCH -t 24:00:00
#SBATCH --mail-user=sam.su@utexas.edu
#SBATCH --mail-type=ALL
#SBATCH -A CCR24018

# Run the tasks
for i in {128..191}; do
  start=$((39 * i))
  end=$((start + 39))
  srun --nodes=1 --ntasks=1 --exclusive --job-name=task$i bash -c "
    source /work/06782/ysu707/ls6/conda/etc/profile.d/conda.sh
    conda activate wrapperbox
    cd /work/06782/ysu707/ls6/WrapperBox/MinimalSubsetToFlipPredictions
    python3 evaluate_main.py --config config/esnli/evaluate_yang_fast/${start}_${end}.json
  " &
done

wait
