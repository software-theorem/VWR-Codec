import cv2
import os
import glob
import numpy as np
from imwatermark import WatermarkEncoder

def process_single_video(encoder, input_path, output_path):
    """
    Function to process a single video.
    Note: This uses the pre-initialized 'encoder' object to save time.
    """
    
    # 1. Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    # Retrieve video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 2. Setup VideoWriter for output
    # 'mp4v' is a standard codec for MP4 containers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # End of video

        # --- Watermarking Process ---
        try:
            # Convert BGR (OpenCV default) -> RGB (Model requirement)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Embed watermark (using the passed encoder instance)
            # Note: 'rivaGan' is case-sensitive
            watermarked_rgb = encoder.encode(frame_rgb, 'rivaGan')
            
            # Convert RGB -> BGR for saving
            watermarked_frame = cv2.cvtColor(watermarked_rgb, cv2.COLOR_RGB2BGR)
            
            out.write(watermarked_frame)
        except Exception as e:
            print(f"Error processing frame: {e}")
            break
        # ----------------------------

        frame_count += 1
        # Print progress every 50 frames to reduce console spam
        if frame_count % 50 == 0:
            print(f"    -> Progress: {frame_count}/{total_frames} frames", end='\r')

    # Release resources
    cap.release()
    out.release()
    print(f"    -> Done! Saved to: {output_path}")

def batch_process():
    # ================= Configuration Area =================
    # 1. Input Folder (Directory containing your 100 videos)
    input_folder = "/home/david/videomarking/Video-Signature-main/output/videos/modelscope/original/videos/"
    
    # 2. Output Folder (Where processed videos will be saved)
    output_folder = "output_rivagan_batch"
    
    # 3. Watermark Data (Must be a 32-bit binary string)
    watermark_bits = '11001100110011001100110011001100'
    # ======================================================

    # Check if output directory exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # === Step 1: Initialize Model (Do this only ONCE!) ===
    print("Initializing RivaGAN model (this happens only once)...")
    encoder = WatermarkEncoder()
    encoder.set_watermark('bits', watermark_bits)
    
    # Load the model (empty parameters automatically load available models)
    encoder.loadModel() 


    # === Step 2: Get all video files ===
    # Using glob to find all .mp4 files in the input folder
    video_files = glob.glob(os.path.join(input_folder, "*.mp4"))
    
    total_files = len(video_files)
    print(f"Found {total_files} video files. Starting batch processing...")

    # === Step 3: Loop through each video ===
    for i, video_path in enumerate(video_files):
        filename = os.path.basename(video_path)
        # Add 'wm_' prefix to output filenames
        output_path = os.path.join(output_folder, f"wm_{filename}")
        
        print(f"\n[{i+1}/{total_files}] Processing: {filename}")
        
        try:
            # Call the processing function, passing the ready-to-use encoder
            process_single_video(encoder, video_path, output_path)
        except Exception as e:
            print(f"Unknown error processing {filename}: {e}")

    print("\n\nAll videos processed successfully!")

if __name__ == "__main__":
    batch_process()