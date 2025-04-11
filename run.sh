#!/bin/bash

# Check if both parameters are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <start_number> <end_number>"
    exit 1
fi

# Assign parameters to variables
start=$1
end=$2

# Validate parameters are integers
if ! [[ "$start" =~ ^[0-9]+$ ]] || ! [[ "$end" =~ ^[0-9]+$ ]]; then
    echo "Error: Both parameters must be integers"
    exit 1
fi

# Handle the case where start is greater than end
if [ $start -gt $end ]; then
    echo "Running experiments in descending order from $start to $end..."
    i=$start
    while [ $i -ge $end ]; do
        echo "Starting experiment $i..."
        python experiment.py "$i"
        echo "Completed experiment $i"
        i=$((i - 1))
    done
else
    echo "Running experiments in ascending order from $start to $end..."
    i=$start
    while [ $i -le $end ]; do
        echo "Starting experiment $i..."
        python experiment.py "$i"
        echo "Completed experiment $i"
        i=$((i + 1))
    done
fi

echo "All experiments completed!"