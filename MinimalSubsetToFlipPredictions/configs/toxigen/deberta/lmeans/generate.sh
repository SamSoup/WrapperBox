#!/bin/bash
nodes=$1
total=$2
# Calculate the ceiling of total / nodes
if [ $((total % nodes)) -ne 0 ]; then
  incre=$((total / nodes + 1))
else
  incre=$((total / nodes))
fi

# Base JSON configuration
base_config='{
    "dataset": "toxigen",
    "model": "deberta_large",
    "seed": 42,
    "pooler": "mean_with_attention",
    "layer": 24,
    "wrapper_name": "LMeans",
    "output_dir": "MinimalSubset/Toxigen/LMeans",
    "iterative_threshold": 100,
    "splits": 100,
    "idx_start": __START__,
    "idx_end": __END__
}'

# Function to generate JSON config file
generate_config() {
  start=$1
  end=$((start + $incre))
  filename="${start}_${end}.json"

  echo "$base_config" | sed -e "s/__START__/${start}/" -e "s/__END__/${end}/" > $filename
}

# Generate config files
for ((i = 0; i < $total; i += $incre)); do
  generate_config $i
done
