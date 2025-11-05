#!/bin/bash

# Dataverse base URL
BASE_URL="https://dataverse.uclouvain.be"

# Dataset DOI
DOI="doi:10.14428/DVN/DEQCIM"

# Metadata file
METADATA_FILE="dataset.json"

# Fetch dataset metadata if not already downloaded
if [ ! -f "$METADATA_FILE" ]; then
    echo "Fetching dataset metadata..."
    curl -s "$BASE_URL/api/datasets/:persistentId?persistentId=$DOI" -o "$METADATA_FILE"
else
    echo "Using cached metadata file: $METADATA_FILE"
fi

# Parse and download each file
echo "Starting download..."
jq -c '.data.latestVersion.files[]' "$METADATA_FILE" | while read file; do
    FILE_ID=$(echo "$file" | jq '.dataFile.id')
    FILENAME=$(echo "$file" | jq -r '.dataFile.filename')

    # Skip file if it already exists
    if [ -f "$FILENAME" ]; then
        echo "Skipping existing file: $FILENAME"
        continue
    fi

    echo "Downloading: $FILENAME (ID: $FILE_ID)"
    curl -L -C - -o "$FILENAME" "$BASE_URL/api/access/datafile/$FILE_ID"

    # Check if download was successful
    if [ $? -eq 0 ]; then
        echo "Finished: $FILENAME"
    else
        echo "Error downloading: $FILENAME — will retry next time"
    fi
done

echo "All files processed."
