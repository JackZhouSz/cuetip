#!/bin/bash

# The command to run and monitor
CMD="python collect_results.py && python run_seq_exp.py --num_trials 50 --splits 5 --index $1"

echo "Running command: $CMD"

# Function to run the command and monitor its output
monitor_command() {
    # Create a temporary file for output
    tmp_file=$(mktemp)
    
    while true; do
        # Start the command in background and redirect output to temp file
        eval "$CMD" > "$tmp_file" 2>&1 &
        cmd_pid=$!
        
        # Start tail in background to show output in real-time
        tail -f "$tmp_file" &
        tail_pid=$!
        
        # Initialize last modification time
        last_mod=$(stat -c %Y "$tmp_file")
        
        while kill -0 $cmd_pid 2>/dev/null; do
            # Check current modification time
            current_mod=$(stat -c %Y "$tmp_file")
            
            # Calculate time difference
            time_diff=$((current_mod - last_mod))
            
            # If no update in 5 minutes (300 seconds), kill and restart
            if [ $time_diff -gt 300 ]; then
                echo "$(date): No output for 5 minutes. Restarting command..."
                kill -9 $cmd_pid
                kill -9 $tail_pid
                break
            fi
            
            # Update last modification time
            last_mod=$current_mod
            
            # Sleep for a bit to avoid excessive CPU usage
            sleep 10
        done
        
        # Small delay before restarting
        sleep 1
    done
    
    # Cleanup
    rm -f "$tmp_file"
}

# Handle script termination
trap 'jobs -p | xargs kill -9 2>/dev/null' EXIT

# Start monitoring
monitor_command