import os
import sys
import torch
import torchvision
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd 
import warnings
import av  

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

# Import VideoSeal model
import videoseal
from videoseal.evals.metrics import bit_accuracy

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

def load_message_from_file(txt_path, nbits):
    """Load original message from txt file"""
    try:
        if not os.path.exists(txt_path):
            return None
            
        with open(txt_path, 'r') as f:
            content = f.read().strip()
        
        # Clean string
        content = content.replace('\n', '').replace(' ', '')
        
        # Convert to list
        msg_list = [int(c) for c in content]
        
        # Handle length mismatch robustly
        if len(msg_list) < nbits:
            # If too short, we can't evaluate accurately. Return None to skip.
            return None
        
        # Take first nbits
        msg_list = msg_list[:nbits]
            
        return torch.tensor(msg_list).float().unsqueeze(0) 
    except Exception as e:
        print(f"[Error] Failed to load message {txt_path}: {e}")
        return None

def read_video_ignoring_audio(video_path):
    """
    """
    container = None
    try:
        container = av.open(video_path)
        if not container.streams.video:
            return torch.empty(0)
            
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO' 
        
        frames = []
        for frame in container.decode(stream):
            frames.append(frame.to_rgb().to_ndarray())
            
        if not frames:
            return torch.empty(0)
            
        video_tensor = torch.from_numpy(np.stack(frames))
        return video_tensor

    except Exception as e:
        raise RuntimeError(f"PyAV failed to read video: {e}")
    finally:
        if container:
            container.close()

def process_video_in_chunks(model, video_path, device, chunk_size=16, interpolation=None):
    """
    Reads video and runs detection in chunks to avoid OOM and dimension errors.
    Returns: Average logits (soft aggregation) across all frames.
    """
    try:
        if not os.path.exists(video_path):
            return None
            
        # 1. Read video using custom PyAV function to ignore Audio errors
        # Returns (T, H, W, C) in [0, 255]
        # frames_uint8, _, _ = torchvision.io.read_video(video_path, pts_unit='sec', output_format="THWC") # OLD Code
        frames_uint8 = read_video_ignoring_audio(video_path)
        
        if frames_uint8.shape[0] == 0:
            return None

        total_frames = frames_uint8.shape[0]
        all_preds = []

        # 2. Process in chunks
        for i in range(0, total_frames, chunk_size):
            # Extract chunk
            chunk = frames_uint8[i : i + chunk_size]
            
            # Normalize to [0, 1] and Float
            chunk = chunk.float() / 255.0
            
            # Permute to (Batch, Channel, Height, Width) where Batch = Chunk Size
            # (N, H, W, C) -> (N, C, H, W)
            chunk = chunk.permute(0, 3, 1, 2)
            
            chunk = chunk.to(device)

            # 3. Run Detection on this batch of "images"
            # We use is_video=False because we are feeding a batch of frames (images)
            # This guarantees compatibility with the underlying 4D interpolate function
            with torch.no_grad():
                res = model.detect(chunk, is_video=False, interpolation=interpolation)
                
                # res['preds'] is usually (Batch, 1 + K) or (Batch, K)
                batch_preds = res['preds']
                
                # If first column is detection score, remove it
                if batch_preds.shape[-1] > 256: # Assuming nbits roughly 256
                      batch_preds = batch_preds[:, 1:]
                
                all_preds.append(batch_preds.cpu())

        if not all_preds:
            return None

        # 4. Aggregate all predictions
        # Concatenate all chunks along batch dimension -> (Total_Frames, K)
        full_preds = torch.cat(all_preds, dim=0)
        
        # Soft Aggregation: Average the logits/probs across all frames
        # This is equivalent to "video_aggregation='avg'"
        avg_preds = full_preds.mean(dim=0, keepdim=True) # (1, K)
        
        return avg_preds.to(device)

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"[Error] OOM processing {os.path.basename(video_path)}")
            torch.cuda.empty_cache()
        else:
            print(f"[Error] Runtime error on {os.path.basename(video_path)}: {e}")
        return None
    except Exception as e:
        print(f"[Error] Failed to process {os.path.basename(video_path)}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_folder', type=str, required=True, help='Root folder containing attacked videos')
    parser.add_argument('--msg_folder', type=str, required=True, help='Folder containing original watermark messages (.txt)')
    parser.add_argument('--nbits', type=int, default=256, help='Number of watermark bits') 
    parser.add_argument('--checkpoint', type=str, default="videoseal", help='Model name')
    parser.add_argument('--output_csv', type=str, default="evaluation_results.csv", help='Path to save the result CSV')
    parser.add_argument('--chunk_size', type=int, default=32, help='Number of frames to process at once (affects VRAM)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model: {args.checkpoint}...")
    try:
        model = videoseal.load(args.checkpoint)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Define interpolation strategy
    interpolation = {
        "mode": "bilinear", 
        "align_corners": False, 
        "antialias": True
    }

    # Initialize CSV
    if not os.path.exists(args.output_csv):
        init_df = pd.DataFrame(columns=['Codec', 'Level', 'Count', 'Bit_Accuracy'])
        init_df.to_csv(args.output_csv, index=False)
        print(f"Created new CSV file: {args.output_csv}")
    else:
        print(f"Appending to existing CSV file: {args.output_csv}")

    print(f"Starting evaluation by folder...")
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
        error_count = 0 
        
        pbar = tqdm(video_files, desc=f"Processing {codec_name}/{level_name}", leave=False)
        
        for vid_filename in pbar:
            vid_path = os.path.join(root, vid_filename)
            
            # --- Message Matching ---
            vid_name_no_ext = os.path.splitext(vid_filename)[0]
            possible_names = [
                vid_name_no_ext.replace("all_videos_", "") + ".txt",
                vid_name_no_ext + ".txt"
            ]
            # Try splitting by underscore for attacked names (e.g. vid_crf23 -> vid.txt)
            parts = vid_name_no_ext.split('_')
            if len(parts) > 1:
                 possible_names.append("_".join(parts[:-1]) + ".txt") # remove last suffix
                 possible_names.append(parts[0] + ".txt") # first part only
            
            if "video" in vid_name_no_ext:
                 nums = "".join(filter(str.isdigit, vid_name_no_ext))
                 if nums: possible_names.append("video" + nums + ".txt")

            txt_path = None
            for name in possible_names:
                temp_path = os.path.join(args.msg_folder, name)
                if os.path.exists(temp_path):
                    txt_path = temp_path
                    break
            
            if txt_path is None:
                if error_count < 1: 
                    tqdm.write(f"[Warning] Skip: Cannot find txt for {vid_filename}")
                error_count += 1
                continue

            # --- Load Message ---
            target_msg = load_message_from_file(txt_path, args.nbits)
            if target_msg is None:
                continue
            target_msg = target_msg.to(device)

            # --- Process Video (Fixed Logic) ---
            # aggregated_preds will be (1, K)
            aggregated_preds = process_video_in_chunks(
                model, 
                vid_path, 
                device, 
                chunk_size=args.chunk_size, 
                interpolation=interpolation
            )
            
            if aggregated_preds is None:
                continue
                
            # --- Calculate Accuracy ---
            acc = bit_accuracy(aggregated_preds, target_msg).item()
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