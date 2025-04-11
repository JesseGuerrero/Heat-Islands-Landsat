#!/bin/bash

# Check if all parameters are provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 <start_number> <end_number> <batch_size>"
    exit 1
fi

# Assign parameters to variables
start=$1
end=$2
batch_size=$3

# Validate parameters are integers
if ! [[ "$start" =~ ^[0-9]+$ ]] || ! [[ "$end" =~ ^[0-9]+$ ]] || ! [[ "$batch_size" =~ ^[0-9]+$ ]]; then
    echo "Error: All parameters must be integers"
    exit 1
fi

# Check if batch size is valid
if [ $batch_size -le 0 ]; then
    echo "Error: Batch size must be greater than 0"
    exit 1
fi

# Handle the case where start is greater than end
if [ $start -gt $end ]; then
    echo "Running experiments in descending order from $start to $end with batch size $batch_size..."
    i=$start
    while [ $i -ge $end ]; do
        batch_end=$((i - batch_size + 1))
        if [ $batch_end -lt $end ]; then
            batch_end=$end
        fi
        
        echo "Starting batch experiments from $i to $batch_end..."
        j=$i
        while [ $j -ge $batch_end ]; do
            echo "Starting experiment $j..."
            python experiment.py "$j"
            echo "Completed experiment $j"
            j=$((j - 1))
        done
        echo "Completed batch $i to $batch_end"
        
        i=$((batch_end - 1))
    done
else
    echo "Running experiments in ascending order from $start to $end with batch size $batch_size..."
    i=$start
    while [ $i -le $end ]; do
        batch_end=$((i + batch_size - 1))
        if [ $batch_end -gt $end ]; then
            batch_end=$end
        fi
        
        echo "Starting batch experiments from $i to $batch_end..."
        j=$i
        while [ $j -le $batch_end ]; do
            echo "Starting experiment $j..."
            python experiment.py "$j"
            echo "Completed experiment $j"
            j=$((j + 1))
        done
        echo "Completed batch $i to $batch_end"
        
        i=$((batch_end + 1))
    done
fi

echo "All experiments completed!"