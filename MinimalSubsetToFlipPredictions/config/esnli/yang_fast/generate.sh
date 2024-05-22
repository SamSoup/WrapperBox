#!/bin/bash
incre=$1

# Base JSON configuration
base_config='{
    "dataset": "esnli",
    "model": "deberta-large",
    "seed": 42,
    "pooler": "mean_with_attention",
    "layer": 24,
    "do_yang2023": true,
    "algorithm_type": "fast",
    "output_dir": "MinimalSubset/yang_fast/__START__to__END__",
    "idx_start": __START__,
    "idx_end": __END__
}'

# Function to generate JSON config file
generate_config() {
  start=$1
  end=$((start + $incre))
  filename="deberta_mean_with_attention_yang_fast_${start}_${end}.json"

  echo "$base_config" | sed -e "s/__START__/${start}/" -e "s/__END__/${end}/" > $filename
}

# Generate config files
for ((i = 0; i < 9900; i += $incre)); do
  generate_config $i
done
