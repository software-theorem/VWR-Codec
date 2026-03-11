#!/bin/bash

# VideoShield Master Attack Script 

# 1. Input root directory
INPUT_ROOT_DIR="data/watermarked/"
# 2. Output directory
OUTPUT_ROOT_DIR="results/"
# 3. Report file path
REPORT_FILE="results/"

if [ -z "$INPUT_ROOT_DIR" ] || [ -z "$OUTPUT_ROOT_DIR" ]; then
    echo "Usage: ./run_all.sh <input_root_dir> <output_root_dir>"
    exit 1
fi

WORKER_SCRIPT="./universal_worker_new.sh" 

if [ ! -f "$WORKER_SCRIPT" ]; then
    echo "Error: Cannot find $WORKER_SCRIPT in current directory."
    exit 1
fi

chmod +x "$WORKER_SCRIPT"


echo "Calculating total files in input directory..."

TOTAL_FILES=$(find "$INPUT_ROOT_DIR" -type f \( -name "*.mp4" -o -name "*.mov" -o -name "*.mkv" -o -name "*.webm" \) | wc -l)

if [ "$TOTAL_FILES" -eq 0 ]; then
    echo "Error: No video files found in input directory: $INPUT_ROOT_DIR"
    exit 1
fi
echo "Total files to process per attack: $TOTAL_FILES"


echo "Attack Benchmark Report - $(date)" > "$REPORT_FILE"
printf "%-10s | %-15s | %-12s | %-12s\n" "Codec" "Level" "Total Time" "Avg/Video" >> "$REPORT_FILE"
echo "-------------------------------------------------------------" >> "$REPORT_FILE"


declare -a CONFIGS=(
    "h264|Level1_Light|-crf 18 -preset fast|mp4"
    "h264|Level2_Low|-crf 23 -preset fast|mp4"
    "h264|Level3_Mid|-crf 28 -preset fast|mp4"
    "h264|Level4_High|-crf 35 -preset fast|mp4"
    "h264|Level5_Extreme|-crf 45 -preset fast|mp4"

    "h265|Level1_Light|-crf 20 -preset fast|mp4"
    "h265|Level2_Low|-crf 26 -preset fast|mp4"
    "h265|Level3_Mid|-crf 32 -preset fast|mp4"
    "h265|Level4_High|-crf 40 -preset fast|mp4"
    "h265|Level5_Extreme|-crf 50 -preset fast|mp4"

    "vp9|Level1_Light|-b:v 0 -crf 15|webm"
    "vp9|Level2_Low|-b:v 0 -crf 25|webm"
    "vp9|Level3_Mid|-b:v 0 -crf 35|webm"
    "vp9|Level4_High|-b:v 0 -crf 45|webm"
    "vp9|Level5_Extreme|-b:v 0 -crf 55|webm"

    "av1|Level1_Light|-preset 6 -crf 20 -an|webm"
    "av1|Level2_Low |-preset 6 -crf 32 -an|webm"
    "av1|Level3_Mid |-preset 6 -crf 45 -an|webm"
    "av1|Level4_High |-preset 6 -crf 55 -an|webm"
    "av1|Level5_Extreme|-preset 6 -crf 63 -an|webm"

    "prores|Level1_4444|-profile:v 4|mov"
    "prores|Level2_HQ|-profile:v 3|mov"
    "prores|Level3_Std|-profile:v 2|mov"
    "prores|Level4_LT|-profile:v 1|mov"
    "prores|Level5_Proxy|-profile:v 0|mov"

    "dnxhd|Level1_444|-profile:v dnxhr_444 -pix_fmt yuv444p10le|mov"
    "dnxhd|Level2_HQX|-profile:v dnxhr_hqx -pix_fmt yuv422p10le|mov"
    "dnxhd|Level3_HQ|-profile:v dnxhr_hq -pix_fmt yuv422p|mov"
    "dnxhd|Level4_SQ|-profile:v dnxhr_sq -pix_fmt yuv422p|mov"
    "dnxhd|Level5_LB|-profile:v dnxhr_lb -pix_fmt yuv422p|mov"
)


total_steps=${#CONFIGS[@]}
current_step=0

echo "Starting Comprehensive Robustness Test..."
echo "Input Directory: $INPUT_ROOT_DIR"
echo "Total Configurations: $total_steps"
echo "--------------------------------------------------------"

for config in "${CONFIGS[@]}"; do
    ((current_step++))
    
    IFS='|' read -r CODEC LEVEL_NAME PARAMS EXT <<< "$config"
    
    CODEC=$(echo "$CODEC" | xargs)
    LEVEL_NAME=$(echo "$LEVEL_NAME" | xargs)
    PARAMS=$(echo "$PARAMS" | xargs)
    EXT=$(echo "$EXT" | xargs)

    TARGET_OUTPUT_DIR="$OUTPUT_ROOT_DIR/$CODEC/$LEVEL_NAME"
    
    echo ">>> [Task $current_step/$total_steps] Running $CODEC - $LEVEL_NAME"
    echo "    Output Dir: $TARGET_OUTPUT_DIR"
    echo "    Format: .$EXT"
    
    start_time=$(date +%s)

    "$WORKER_SCRIPT" "$INPUT_ROOT_DIR" "$TARGET_OUTPUT_DIR" "$CODEC" "$PARAMS" "$EXT"
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    if [ "$duration" -eq 0 ]; then duration=1; fi

    avg_time=$(awk "BEGIN {printf \"%.2f\", $duration / $TOTAL_FILES}")

    echo "    Finished in: ${duration}s | Average: ${avg_time}s per video"
    
    printf "%-10s | %-15s | %-12s | %-12s\n" "$CODEC" "$LEVEL_NAME" "${duration}s" "${avg_time}s" >> "$REPORT_FILE"

    echo "--------------------------------------------------------"
done

echo "ALL ATTACKS COMPLETED SUCCESSFULLY."
echo "View performance report at: $REPORT_FILE"