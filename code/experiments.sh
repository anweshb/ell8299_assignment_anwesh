#!/bin/bash

# Grid search parameters
sequence_lengths=(64 128)
num_layers=(3 6)
num_heads=(8 16)
learning_rates=(0.0003)
weight_decays=(0.01)

# Create log directory
log_dir="/home/anwesh/scratch/ELL8299 Project/logs"
mkdir -p "$log_dir"

# Get current timestamp for run identification
timestamp=$(date +%Y%m%d_%H%M%S)

# Check if CUDA device 1 is available
# if ! nvidia-smi -L | grep -q "GPU 1:"; then
#     echo "Error: CUDA device 1 is not available"
#     exit 1
# fi


# "seq64_l3_h8_lr0.0003_wd0.0"
# "seq64_l3_h16_lr0.0003_wd0.0"
# "seq64_l3_h8_lr0.0003_wd0.01"


skip_configs=(

    # You can add more configurations here, e.g.:
    # "seq128_l6_h16_lr0.0003_wd0.01"
)


# Force Python to flush output immediately
export PYTHONUNBUFFERED=1

for seq_len in "${sequence_lengths[@]}"; do
    for layers in "${num_layers[@]}"; do
        for heads in "${num_heads[@]}"; do
            for lr in "${learning_rates[@]}"; do
                for wd_raw in "${weight_decays[@]}"; do
                    # Normalize weight_decay for consistent run_name (e.g., 0.0 to 0.0)
                    # This ensures "0" matches "0.0" if both are used as raw inputs
                    wd=$(echo "$wd_raw" | awk '{printf "%.4f", $1}')
                    
                    # Create a unique name for this run
                    run_name="seq${seq_len}_l${layers}_h${heads}_lr${lr}_wd${wd_raw}_high_patience"
                    log_file="${log_dir}/${timestamp}_${run_name}.log"
                    
                    # =================================================
                    # REPLACED: Generalized Skip Check
                    # Uses a single loop and a temporary check variable
                    # =================================================
                    SKIP=0
                    for skip_name in "${skip_configs[@]}"; do
                        if [[ "$run_name" == "$skip_name" ]]; then
                            SKIP=1
                            break
                        fi
                    done
                    
                    if [[ "$SKIP" -eq 1 ]]; then
                       echo "Skipping already completed run: $run_name"
                       continue
                    fi
                    # =================================================
                    
                    echo "Starting run: $run_name"
                    echo "Logging to: $log_file"
                    
                    # Run with unbuffered output and use script to capture terminal output
                    script -q -c "python -u ./train.py \
                        --seq_len $seq_len \
                        --num_layers $layers \
                        --num_heads $heads \
                        --lr $lr \
                        --weight_decay $wd_raw \
                        --num_epochs 100 \
                        --wandb_project decoder-transformer-multiple_models" "$log_file"
                    
                    # Optional: add a small delay between runs
                    sleep 5
                done
            done
        done
    done
done

echo "Grid search completed!"