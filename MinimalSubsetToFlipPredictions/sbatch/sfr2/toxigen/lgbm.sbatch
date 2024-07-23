#!/bin/bash
#SBATCH -J sfr2_toxigen_minimal_subset_lgbm
#SBATCH -o sfr2_toxigen_minimal_subset_lgbm.o%j
#SBATCH -e sfr2_toxigen_minimal_subset_lgbm.e%j
#SBATCH -p skx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH --mail-user=sam.su@utexas.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH -A CCR24018
source /work/06782/ysu707/ls6/conda/etc/profile.d/conda.sh
conda activate wrapperbox
cd /work/06782/ysu707/ls6/WrapperBox/MinimalSubsetToFlipPredictions
python3 main.py --config configs/toxigen/sfr2/lgbm.json
