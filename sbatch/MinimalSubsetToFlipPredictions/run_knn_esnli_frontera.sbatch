#!/bin/bash
#SBATCH -J deberta_large_knn_minimal_subset_esnli_frontera
#SBATCH -o deberta_large_knn_minimal_subset_esnli_frontera.o%j
#SBATCH -e deberta_large_knn_minimal_subset_esnli_frontera.e%j
#SBATCH -p small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 42:00:00
#SBATCH --mail-user=sam.su@utexas.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH -A CCR24018
source /work/06782/ysu707/ls6/conda/etc/profile.d/conda.sh
conda activate wrapperbox
cd /work/06782/ysu707/ls6/WrapperBox/MinimalSubsetToFlipPredictions
python3 main.py --config config/esnli/deberta_mean_with_attention_knn.json
