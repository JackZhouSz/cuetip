#!/bin/bash

# This script demonstrates how to use visualise_gif.py to create motion trail images from GIFs
# The script takes either a single GIF file or a directory containing GIFs and creates motion trail images
#
# Usage:
#   ./create_motion_trails.sh <input_file_or_directory>
#
# Examples:
#   ./create_motion_trails.sh examples/shot.gif    # Process single file
#   ./create_motion_trails.sh examples/            # Process all GIFs in directory
#
# Parameters for motion trails can be adjusted:
#   --alpha (-a): Controls the transparency of the motion trail (default: 0.8)
#   --sigma (-s): Controls the blur effect of the trail (default: 5)
#   Higher alpha = more visible trail
#   Higher sigma = more blurred trail

# Check if input is provided
if [ $# -eq 0 ]; then
    echo "Error: Please provide an input file or directory"
    echo "Usage: $0 <input_file_or_directory>"
    exit 1
fi

input_path="$1"

# Function to process a single GIF file
process_gif() {
    local gif_file="$1"
    # Create output filename by replacing .gif with _motion_trail.png
    local output_file="${gif_file%.gif}_motion_trail.png"
    
    echo "Processing: $gif_file"
    echo "Creating motion trail: $output_file"
    
    # Run the Python script with default parameters
    python visualise_gif.py "$gif_file" -o "$output_file"
    
    # Example with custom parameters (commented out)
    # For stronger trails, uncomment this and comment out the line above:
    # python visualise_gif.py "$gif_file" -o "$output_file" -a 0.9 -s 3
}

# Check if input is a file or directory
if [ -f "$input_path" ]; then
    # Check if it's a GIF file
    if [[ "$input_path" == *.gif ]]; then
        process_gif "$input_path"
    else
        echo "Error: Input file must be a GIF file"
        exit 1
    fi
elif [ -d "$input_path" ]; then
    # Process all GIF files in the input directory
    find "$input_path" -name "*.gif" | while read -r gif_file; do
        process_gif "$gif_file"
    done
else
    echo "Error: $input_path does not exist or is neither a file nor directory"
    exit 1
fi

echo "Motion trail creation complete!"