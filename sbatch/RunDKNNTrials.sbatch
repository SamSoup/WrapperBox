#!/bin/bash
#SBATCH -J deberta_large_all_trials_toxigen_part_2
#SBATCH -o deberta_large_all_trials_toxigen_part_2.o%j
#SBATCH -e deberta_large_all_trials_toxigen_part_2.e%j
#SBATCH -p gtx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --mail-user=sam.su@utexas.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH -A CCR21016
source /work/06782/ysu707/ls6/conda/etc/profile.d/conda.sh
conda activate DkNN
cd /work/06782/ysu707/ls6/DkNN
# ./RunDKNNTrials.sh ./trial_configurations/toxigen/deberta-large-1-new/ 0
./RunDKNNTrialsRange.sh 168 335 ./trial_configurations/toxigen/deberta-large-1-new/