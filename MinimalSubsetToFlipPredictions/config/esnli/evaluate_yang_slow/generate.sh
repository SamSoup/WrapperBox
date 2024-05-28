#!/bin/bash
incre=$1

# Base JSON configuration
base_config='{
    "dataset": "esnli",
    "subsets_filename": "esnli_deberta_large_yang2023_slow.pickle",
    "model": "deberta_large",
    "seed": 42,
    "pooler": "mean_with_attention",
    "layer": 24,
    "do_yang2023": true,
    "algorithm_type": "slow",
    "output_dir": "MinimalSubset/evaluate/yang_slow/",
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
for ((i = 0; i < 9824; i += $incre)); do
  generate_config $i
done
