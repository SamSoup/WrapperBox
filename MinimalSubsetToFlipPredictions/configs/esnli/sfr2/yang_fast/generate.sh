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
    "dataset": "esnli_subset",
    "model": "sfr2_instruct_alt",
    "seed": 42,
    "output_dir": "./results/sfr2/esnli_subset/yang_fast",
    "load_sentence_transformer_embedding": true,
    "load_sentence_transformer_wrapper": true,
    "thresh": 0.5,
    "l2": 1,
    "do_yang2023": true,
    "wrapper_name": "LogisticRegression",
    "algorithm_type": "fast",
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
