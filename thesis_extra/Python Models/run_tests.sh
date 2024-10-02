#!/bin/bash

# Number of iterations
I=1 # Set the number of iterations you want

# Initialize log files
echo "Iteration,C_Memory,C_Time,Python_Memory,Python_Time" > results.csv

# Loop to run the C and Python scripts I times
for ((i=1; i<=I; i++))
do
    echo "Iteration $i"
    
    # Recompile the C program
    gcc -O3 -march=native -o multihead multihead.c -lblas -lm

    # Check if compilation was successful
    # if [ $? -ne 0 ]; then
    #     echo "Compilation failed at iteration $i"
    #     exit 1
    # fi

    # Run the C program and capture its output
    ./multihead > c_output.txt

    # Extract memory usage and time taken from the C program's output
    c_memory=$(grep "Memory usage:" c_output.txt | grep -oP '\d+')
    c_time=$(grep "Time taken for encoder-block:" c_output.txt | grep -oP '\d+\.\d+')

    # Run the Python script and capture its output
    python3 head.py > py_output.txt

    # Extract memory usage and time taken from the Python script's output
    py_memory=$(grep "Memory usage:" py_output.txt | grep -oP '\d+\.\d+')
    py_time=$(grep "Time taken:" py_output.txt | grep -oP '\d+\.\d+')
    py_result=$(grep "Tensors for MHA" py_output.txt | grep -oP '\d+\.\d+')

    # Log the data to the results file
    echo "$i,$c_memory,$c_time,$py_memory,$py_time" >> results.csv

    echo "C Program - Memory: ${c_memory} KB, Time: ${c_time} seconds"
    echo "Python Script - Memory: ${py_memory} MB, Time: ${py_time} seconds"
    echo "${py_result}"

    echo "================================="
done
