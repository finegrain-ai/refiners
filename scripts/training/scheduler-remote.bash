#!/bin/bash

# Path to the directory containing the config files
config_dir="./configs/scheduled-remote"
prefix="finetune-color-palette-schedule-"
script="./scripts/training/finetune-ldm-color-palette.py"
# Log file path
log_file="./tmp/schedule-log.txt"

# Ensure the log directory exists
mkdir -p "$(dirname "$log_file")"

# Write start time to the log file
echo "Script started at $(date)" >> "$log_file"
echo "Config files to be processed:" >> "$log_file"
ls "$config_dir"/"$prefix"*.toml >> "$log_file"

# Loop through each config file matching the pattern and run the script
for config in "$config_dir"/"$prefix"*.toml
do
    echo "Running : $script $config"
    start_time=$(date +%s)

    if rye run python $script $config; then
        status="Success"
    else
        status="Failure"
    fi

    end_time=$(date +%s)
    duration=$((end_time - start_time))

    echo "Script run with $config finished. Time spent: $duration seconds. Status: $status" >> "$log_file"
done

# Write end time to the log file
echo "Script ended at $(date)" >> "$log_file"

echo "All scripts have been run. Check $log_file for details."
