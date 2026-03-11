import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd 
import warnings
import av  

# Import the library
try:
    from imwatermark import WatermarkDecoder
except ImportError:
    print("[Error] 'invisible-watermark' library not found. Please run 'pip install invisible-watermark'")
    sys.exit(1)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

def calculate_bit_accuracy(decoded_bits, target):
    """
    Calculate bit accuracy.
    """
    # Convert to numpy for comparison
    d = np.array(decoded_bits).flatten()
    t = np.array(target).flatten()
    
    # Handle length mismatch (truncate to shorter)
    min_len = min(len(d), len(t))
    d = d[:min_len]
    t = t[:min_len]

    if min_len == 0:
        return 0.0

    # Calculate accuracy
    correct = (d == t).mean()
    return correct

def parse_target_bits(bit_string):
    """Convert string '110011' to list [1, 1, 0, 0, 1, 1]"""
    try:
        return [int(b) for b in bit_string if b in '01']
    except Exception as e:
        print(f"[Error] Invalid bit string: {e}")
        return None

def read_video_frames(video_path):
    """
    Read video frames using PyAV.
    Returns a list of numpy arrays (H, W, C) in RGB format.
    """
    container = None
    try:
        container = av.open(video_path)
        if not container.streams.video:
            return []
            
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO' 
        
        frames = []
        for frame in container.decode(stream):
            frames.append(frame.to_rgb().to_ndarray())
            
        return frames

    except Exception as e:
        return []
    finally:
        if container:
            container.close()

def decode_video_aggregated(decoder, video_path, nbits):
    """
    Reads video, decodes each frame, and aggregates results (Majority Vote).
    """
    frames = read_video_frames(video_path)
    
    if not frames:
        return None
    
    all_decoded_msgs = []

    # Process each frame
    for frame in frames:
        try:
            bits = decoder.decode(frame, method='rivaGan')
            
            if bits is not None and len(bits) == nbits:
                all_decoded_msgs.append(bits)
        except Exception:
            continue

    if not all_decoded_msgs:
        return None

    # Aggregate: Majority Vote
    all_decoded_msgs = np.array(all_decoded_msgs)
    
    # Average column-wise
    avg_bits = all_decoded_msgs.mean(axis=0)
    
    # Threshold at 0.5
    final_bits = (avg_bits > 0.5).astype(int).tolist()
    
    return final_bits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_folder', type=str, required=True, help='Root folder containing attacked videos')
    parser.add_argument('--target_bits', type=str, required=True, help='The 32-bit watermark string (e.g. 11001100...)')
    parser.add_argument('--nbits', type=int, default=32, help='Number of watermark bits') 
    parser.add_argument('--output_csv', type=str, default="rivagan_results_final.csv", help='Path to save the result CSV')
    args = parser.parse_args()
    
    # --- Parse Target Message ---
    target_msg = parse_target_bits(args.target_bits)
    if target_msg is None or len(target_msg) != args.nbits:
        print(f"[Error] Target bits length ({len(target_msg) if target_msg else 0}) does not match nbits ({args.nbits})")
        return

    print(f"Target Watermark: {args.target_bits}")

    # --- Initialize Watermark Decoder ---
    print(f"Initializing RivaGAN decoder (nbits={args.nbits})...")
    try:
        # 1. Initialize with type 'bits'
        decoder = WatermarkDecoder('bits', args.nbits)
        
        # 2. IMPORTANT FIX: Explicitly load the deep learning model!
        print("Loading RivaGAN model weights...")
        decoder.loadModel() 
        
        print("Decoder loaded successfully.")
    except Exception as e:
        print(f"[Error] Failed to initialize decoder: {e}")
        return

    # Initialize CSV
    if not os.path.exists(args.output_csv):
        init_df = pd.DataFrame(columns=['Codec', 'Level', 'Count', 'Bit_Accuracy'])
        init_df.to_csv(args.output_csv, index=False)
        print(f"Created new CSV file: {args.output_csv}")
    else:
        print(f"Appending to existing CSV file: {args.output_csv}")
    
    print(f"Starting evaluation...")
    print("\n" + "="*75)
    print(f"{'CODEC':<10} | {'LEVEL':<20} | {'COUNT':<6} | {'BIT ACCURACY'}")
    print("="*75)

    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'} 
    total_videos = 0
    
    # Traverse folders
    for root, dirs, files in os.walk(args.video_folder):
        video_files = [f for f in files if os.path.splitext(f)[1].lower() in video_extensions]
        
        if not video_files:
            continue

        level_name = os.path.basename(root)
        codec_dir = os.path.dirname(root)
        codec_name = os.path.basename(codec_dir)
        if level_name == os.path.basename(args.video_folder):
            level_name = "Root"
            codec_name = "None"

        current_folder_accs = []
        
        pbar = tqdm(video_files, desc=f"Processing {codec_name}/{level_name}", leave=False)
        
        for vid_filename in pbar:
            vid_path = os.path.join(root, vid_filename)
            
            # --- Decode Video ---
            predicted_bits = decode_video_aggregated(decoder, vid_path, args.nbits)
            
            if predicted_bits is None:
                continue
                
            # --- Calculate Accuracy ---
            acc = calculate_bit_accuracy(predicted_bits, target_msg)
            current_folder_accs.append(acc)

        
        # Save results for folder
        if current_folder_accs:
            avg_acc = sum(current_folder_accs) / len(current_folder_accs)
            count = len(current_folder_accs)
            total_videos += count
            
            print(f"{codec_name:<10} | {level_name:<20} | {count:<6} | {avg_acc:.4f}")
            
            new_row = {
                'Codec': codec_name,
                'Level': level_name,
                'Count': count,
                'Bit_Accuracy': avg_acc
            }
            pd.DataFrame([new_row]).to_csv(args.output_csv, mode='a', header=False, index=False)
        else:
            if video_files:
                print(f"{codec_name:<10} | {level_name:<20} | 0      | FAILED")

    print("="*75)
    print(f"Finished. Total processed videos: {total_videos}")
    print(f"Results saved to: {os.path.abspath(args.output_csv)}")

if __name__ == "__main__":
    main()