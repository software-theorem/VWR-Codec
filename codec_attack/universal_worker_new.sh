#!/bin/bash

if [ $# -lt 5 ]; then
    echo "Usage: $0 <input_folder> <output_folder> <codec_type> <encode_params> <extension>"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
CODEC_TYPE="$3"
USER_PARAMS="$4" 
OUTPUT_EXT="$5"

if [ -z "$OUTPUT_EXT" ]; then
    echo "Error: Output extension not specified."
    exit 1
fi

if [[ "$OUTPUT_EXT" != .* ]]; then
    FINAL_EXT=".$OUTPUT_EXT"
else
    FINAL_EXT="$OUTPUT_EXT"
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input folder does not exist: $INPUT_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

SCALE_FILTER="-vf scale=512:512,setsar=1"

echo "Starting Batch Processing..."
echo "Mode: $CODEC_TYPE"
echo "Resolution: 512x512 (Forced)"
echo "Output Format: $FINAL_EXT"
echo "--------------------------------------------------------"


case "$CODEC_TYPE" in
    h264)
        BASE_FLAGS="-c:v libx264 -movflags +faststart"
        AUDIO_FLAGS="-c:a aac -b:a 128k"
        ;;
    h265)
        BASE_FLAGS="-c:v libx265 -tag:v hvc1"
        AUDIO_FLAGS="-c:a aac -b:a 128k"
        ;;
    prores)
        BASE_FLAGS="-c:v prores_ks"
        AUDIO_FLAGS="-c:a pcm_s16le -ar 48000"
        ;;
    dnxhd)
        BASE_FLAGS="-c:v dnxhd"
        AUDIO_FLAGS="-c:a pcm_s16le -ar 48000"
        ;;
    vp9)
        BASE_FLAGS="-c:v libvpx-vp9"
        AUDIO_FLAGS="-c:a libopus"
        ;;
    av1)
        BASE_FLAGS="-c:v libsvtav1"
        AUDIO_FLAGS="-c:a aac -b:a 128k"
        ;;
    *)
        echo "Error: Unknown codec type '$CODEC_TYPE'"
        exit 1
        ;;
esac

find "$INPUT_DIR" -type f \( -name "*.mp4" -o -name "*.mov" -o -name "*.mkv" -o -name "*.webm" -o -name "*.avi" \) | while IFS= read -r input_video; do
    
    filename=$(basename -- "$input_video")
    parent_dir_path=$(dirname "$input_video")
    parent_dir_name=$(basename "$parent_dir_path")
    
    filename_no_ext="${filename%.*}"
    
    new_filename="${parent_dir_name}_${filename_no_ext}${FINAL_EXT}"
    output_video="$OUTPUT_DIR/$new_filename"

    echo "Processing: $filename -> 512x512 -> $new_filename"

    
    ffmpeg -v error -i "$input_video" \
        $SCALE_FILTER \
        $BASE_FLAGS \
        $USER_PARAMS \
        $AUDIO_FLAGS \
        -strict -2 \
        -y \
        "$output_video" < /dev/null

    if [ $? -eq 0 ]; then
        echo " -> [OK]"
    else
        echo " -> [FAILED] Error processing $filename"
        if [ -f "$output_video" ] && [ ! -s "$output_video" ]; then
            rm "$output_video"
            echo "    (Deleted empty output file)"
        fi
    fi
done

echo "========================================================"
echo "Batch finished for $CODEC_TYPE."