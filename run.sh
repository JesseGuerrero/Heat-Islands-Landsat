#!/bin/sh
# Run all experiments sequentially
i=14
while [ $i -le 16 ]
do
   echo "Starting experiment $i..."
   python experiment.py "$i"
   echo "Completed experiment $i"
   i=$((i + 1))
done
echo "All experiments completed!"
