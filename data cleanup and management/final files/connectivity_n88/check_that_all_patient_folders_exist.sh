#!/bin/bash

# -------------------------------------------------------------------
# Script Name: check_patient_folders.sh
#
# Description:
#   This script checks if folders corresponding to patient IDs listed
#   in a CSV file exist in the current directory. The folders are
#   expected to be named with 9-digit numeric IDs.
#
# Assumptions:
#   - The CSV file is named `unique_ids.csv`.
#   - The CSV file contains one patient ID per line.
#   - Patient IDs in the CSV file may be fewer than 9 digits;
#     in that case, leading zeros are added to make them 9 digits.
#   - Each patient folder is named exactly as the 9-digit ID.
#   - The script is run in the directory containing the patient folders.
#
# Usage:
#   1. Place this script in the directory containing patient folders.
#   2. Ensure `unique_ids.csv` is present in the same directory.
#   3. Run the script: `bash check_patient_folders.sh`
# -------------------------------------------------------------------

CSV_FILE="unique_ids.csv"

# Check if CSV file exists
if [[ ! -f "$CSV_FILE" ]]; then
  echo "Error: CSV file '$CSV_FILE' not found."
  exit 1
fi

missing_ids=()

while IFS= read -r line || [[ -n "$line" ]]; do
  # Strip whitespace and convert to 9-digit format
  raw_id=$(echo "$line" | tr -d '[:space:]')
  padded_id=$(printf "%07d" "$raw_id")

  # Check if folder exists
  if [[ ! -d "$padded_id" ]]; then
    missing_ids+=("$padded_id")
  fi
done < "$CSV_FILE"

# Output result
if [[ ${#missing_ids[@]} -eq 0 ]]; then
  echo "All IDs in '$CSV_FILE' have corresponding folders."
else
  echo "The following IDs do not have corresponding folders:"
  for id in "${missing_ids[@]}"; do
    echo "$id"
  done
fi
 
