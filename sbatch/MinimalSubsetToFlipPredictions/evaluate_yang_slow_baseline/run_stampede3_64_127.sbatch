#!/bin/bash
#SBATCH -J esnli_evaluate_yang_slow_baseline_multinode_st3_second_batch
#SBATCH -o esnli_evaluate_yang_slow_baseline_multinode_st3_second_batch.o%j
#SBATCH -e esnli_evaluate_yang_slow_baseline_multinode_st3_second_batch.e%j
#SBATCH -p skx
#SBATCH -N 64
#SBATCH -n 64
#SBATCH -t 24:00:00
#SBATCH --mail-user=sam.su@utexas.edu
#SBATCH --mail-type=ALL
#SBATCH -A CCR24018

num_samples=9824
# Run the tasks
for i in {64..127}; do
  start=$((77 * i))
  end=$((start + 77))
  if [ $num_samples -lt $end ]; then
      end=$num_samples
  fi
  srun --nodes=1 --ntasks=1 --exclusive --job-name=task$i bash -c "
    source /work/06782/ysu707/ls6/conda/etc/profile.d/conda.sh
    conda activate wrapperbox
    cd /work/06782/ysu707/ls6/WrapperBox/MinimalSubsetToFlipPredictions
    python3 verify_validity.py --config config/esnli/evaluate_yang_slow_baseline/${start}_${end}.json
  " &
done

wait
