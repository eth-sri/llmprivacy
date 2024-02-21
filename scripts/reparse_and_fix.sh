#!/bin/sh

# Script used for re-parsing and fixing the jsonl files in the reddit dataset (especially non GPT-4 answers with inconsistent format)

FOLDER_A=$1
FOLDER_B=$2
FOLDER_C=$3

# iterate over all files in folder A
# for filename in "$FOLDER_A"/*.jsonl; do
#     # Extract base name to avoid folder part of filename
#     base=$(basename "$filename" .jsonl)

#     # Append _reparse.jsonl to the base name
#     new_base="${base}_reparse.jsonl"

#     # Define the new filename in folder B
#     TARGET="$FOLDER_B/$new_base"

#     # call python script
#     python3 ./src/reddit/normalize.py --in_paths $filename --outpath $TARGET --reparse
# done

# # iterate over all files in folder B
for filename in "$FOLDER_B"/*_reparse.jsonl; do
    # Extract base name to avoid folder part of the filename
    base=$(basename "$filename" _reparse.jsonl) 

    # Append _fix.jsonl to the base name
    new_base="${base}_fix.jsonl" 

    # Define the new filename in folder C
    TARGET="$FOLDER_C/$new_base" 

    # call python script
    python3 ./src/reddit/normalize.py --in_paths $filename --outpath $TARGET --fix
done