#!/bin/bash

# ----------------------------------------------------------------------
# Script Name: check_patient_folders.sh
#
# Description:
# This script checks if folders named after patient IDs exist in the 
# current directory, based on a CSV file (`unique_ids.csv`) containing 
# eligible patient IDs.
#
# Assumptions:
# - Each patient ID in the CSV file is numerical and may be fewer than 9 digits.
# - IDs should be zero-padded on the left to ensure a 9-digit folder name.
# - Folders corresponding to the patient IDs are located in the current working directory.
# - The CSV file (`unique_ids.csv`) is in the same directory where the script is run.
#
# Usage:
# 1. Place this script in the directory with the patient folders and the `unique_ids.csv` file.
# 2. Make the script executable: chmod +x check_patient_folders.sh
# 3. Run the script: ./check_patient_folders.sh
#
# Output:
# - Lists any IDs from the CSV file that do not have a corresponding folder.
# ----------------------------------------------------------------------

# Set the CSV filename
CSV_FILE="unique_ids.csv"

# Check if the CSV file exists
if [[ ! -f "$CSV_FILE" ]]; then
    echo "Error: File '$CSV_FILE' not found in current directory."
    exit 1
fi

# Initialize a flag to track missing folders
missing_flag=0

echo "Checking for missing patient folders..."

# Read CSV line by line
while IFS=, read -r id; do
    # Trim whitespace and pad with leading zeros to 9 digits
    clean_id=$(printf "%07d" "$id")
    
    # Check if the directory exists
    if [[ ! -d "$clean_id" ]]; then
        echo "Missing folder for patient ID: $clean_id"
        missing_flag=1
    fi
done < "$CSV_FILE"

# Summary message
if [[ $missing_flag -eq 0 ]]; then
    echo "All patient folders exist."
else
    echo "Some patient folders are missing. Please check the list above."
fi
 
