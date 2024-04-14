#!/bin/bash

# Data directory
DATA_DIR="data"

# Function to download files from Google Drive using gdown and print progress
download_and_extract_file() {
    file_id=$1

    # download file
    gdown --fuzzy "$file_id" -O "$DATA_DIR/$file_id.tar"

    # extract file
    tar -xvf "$DATA_DIR/$file_id.tar" -C "$DATA_DIR"

    # remove file
    rm "$DATA_DIR/$file_id.tar"
}

# List of Google Drive file IDs
file_ids=(
    "1A2IU8Sgea1h3fYLpYtFb2v7NYdMjvEhU" # Task01 brain tumor
    # "1wEB2I6S6tQBVEPxir8cA5kFB8gTQadYY" # Task02 heart
    # Add more file IDs as needed
)

# Create data directory if it doesn't exist
mkdir -p "$DATA_DIR"

# Download and extract files in parallel
for id in "${file_ids[@]}"; do
    download_and_extract_file "$id"
done

# Wait for all extraction processes to finish
wait

echo "All files downloaded and extracted successfully into $DATA_DIR."
