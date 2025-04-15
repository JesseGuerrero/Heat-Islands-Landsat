#!/bin/bash
# Change to the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"


# Check if all parameters are provided
if [ $# -ne 4 ]; then
    echo "Usage: $0 <start_number> <end_number> <batch_size> <num_gpus>"
    exit 1
fi

# Assign parameters to variables
start=$1
end=$2
batch_size=$3
num_gpus=$4

# Validate parameters are integers
if ! [[ "$start" =~ ^[0-9]+$ ]] || ! [[ "$end" =~ ^[0-9]+$ ]] || ! [[ "$batch_size" =~ ^[0-9]+$ ]] || ! [[ "$num_gpus" =~ ^[0-9]+$ ]]; then
    echo "Error: All parameters must be integers"
    exit 1
fi

# Check if batch size is valid
if [ $batch_size -le 0 ]; then
    echo "Error: Batch size must be greater than 0"
    exit 1
fi

# Check if start number is less than or equal to end
if [ $start -gt $end ]; then
    echo "Error: Start number must be less than or equal to end number"
    exit 1
fi

echo "Running experiments from $start to $end with batch size $batch_size and $num_gpus GPUs..."

# Run for each number in the range
i=$start
while [ $i -le $end ]; do
    echo "Starting experiment $i with batch size $batch_size and $num_gpus GPUs..."
    python experiment.py "$i" "$batch_size" "$num_gpus"
    echo "Completed experiment $i"
    i=$((i + 1))
done

echo "All experiments completed!"