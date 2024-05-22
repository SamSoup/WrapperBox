#!/bin/bash

# Base job name and email
base_job_name="deberta_large_lmeans_subset_esnli"
email="sam.su@utexas.edu"
account="CCR24018"
config_base="config/esnli/lmeans/deberta_mean_with_attention_LMeans"

# Function to generate sbatch script
generate_script() {
  start=$1
  end=$((start + 700))
  suffix=$2
  filename="run_deberta_mean_with_attention_LMeans_${start}_${end}.sbatch"

  cat <<EOL > $filename
#!/bin/bash
#SBATCH -J ${base_job_name}_${start}_${end}_normal_${suffix}
#SBATCH -o ${base_job_name}_${start}_${end}_normal_${suffix}.o%j
#SBATCH -e ${base_job_name}_${start}_${end}_normal_${suffix}.e%j
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH --mail-user=${email}
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH -A ${account}
source /work/06782/ysu707/ls6/conda/etc/profile.d/conda.sh
conda activate wrapperbox
cd /work/06782/ysu707/ls6/WrapperBox/MinimalSubsetToFlipPredictions
python3 main.py --config ${config_base}_${start}_${end}.json
EOL
}

# Generate scripts
suffix=1
for ((i = 0; i < 10500; i += 700)); do
  generate_script $i $suffix
  suffix=$((suffix + 1))
done
