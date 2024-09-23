#!/bin/bash

# Directory to clean (defaults to the current directory if not provided)
DIRECTORY='/Users/leiapei/Downloads/2dshifts/outfiles'

# Remove all .out files within the directory
find "$DIRECTORY" -type f -name "*.out" -exec rm -f {} \;

echo "All .out files in $DIRECTORY have been removed."