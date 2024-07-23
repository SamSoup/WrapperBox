#!/bin/bash
#SBATCH -J sfr2_esnli_subset_minimal_subset_lmeans
#SBATCH -o sfr2_esnli_subset_minimal_subset_lmeans.o%j
#SBATCH -e sfr2_esnli_subset_minimal_subset_lmeans.e%j
#SBATCH -p skx
#SBATCH -N 16
#SBATCH -n 16
#SBATCH -t 48:00:00
#SBATCH --mail-user=sam.su@utexas.edu
#SBATCH --mail-type=ALL
#SBATCH -A CCR24018

num_samples=9824
# Run the tasks
for i in {0..15}; do
  start=$((614 * i))
  end=$((start + 614))
  if [ $num_samples -lt $end ]; then
      end=$num_samples
  fi
  srun --nodes=1 --ntasks=1 --exclusive --job-name=task$i bash -c "
    source /work/06782/ysu707/ls6/conda/etc/profile.d/conda.sh
    conda activate wrapperbox
    cd /work/06782/ysu707/ls6/WrapperBox/MinimalSubsetToFlipPredictions
    python3 main.py --config configs/esnli/sfr2/lmeans/${start}_${end}.json
  " &
done

wait
