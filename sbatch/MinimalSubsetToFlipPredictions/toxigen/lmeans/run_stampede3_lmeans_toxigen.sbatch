#!/bin/bash
#SBATCH -J toxigen_lmeans_multinode
#SBATCH -o toxigen_lmeans_multinode.o%j
#SBATCH -e toxigen_lmeans_multinode.e%j
#SBATCH -p skx
#SBATCH -N 50
#SBATCH -n 50
#SBATCH -t 2:00:00
#SBATCH --mail-user=sam.su@utexas.edu
#SBATCH --mail-type=ALL
#SBATCH -A CCR24018

# Run the tasks
for i in {0..49}; do
  start=$((19 * i))
  end=$((start + 19))
  srun --nodes=1 --ntasks=1 --exclusive --job-name=task$i bash -c "
    source /work/06782/ysu707/ls6/conda/etc/profile.d/conda.sh
    conda activate wrapperbox
    cd /work/06782/ysu707/ls6/WrapperBox/MinimalSubsetToFlipPredictions
    python3 main.py --config config/toxigen/lmeans/${start}_${end}.json
  " &
done

wait
