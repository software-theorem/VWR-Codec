import os
import argparse
import torch
import cv2
import numpy as np
import csv
from tqdm import tqdm
from diffusers import TextToVideoSDPipeline, StableVideoDiffusionPipeline, DDIMInverseScheduler
from utils import transform_video, cv2_to_pil, get_video_latents
from watermark import VideoShield

# [IMPORTANT] Use ImageIO V2 API for maximum compatibility with all codecs
import imageio

def load_video_frames(video_path):
    """
    Loads video frames using ImageIO (ffmpeg plugin).
    
    Features:
    1. Robustness: Reads AV1, VP9, DNxHD, and ProRes files that OpenCV might fail on.
    2. Smart Normalization: Automatically detects if the video is 8-bit (uint8) 
       or 10/12-bit (uint16) and normalizes pixel values to the [0.0, 1.0] float range 
       required by the AI model.
    """
    try:
        if not os.path.exists(video_path):
            return []
            
        # Use get_reader (V2 API) without forcing pixel format.
        # This allows ImageIO to read the native bit-depth (8, 10, or 12 bits).
        reader = imageio.get_reader(video_path, 'ffmpeg')
        
        frames_list = []
        for frame in reader:
            frames_list.append(frame)
        reader.close()
        
        if len(frames_list) == 0:
            return []

        # Check the data type of the first frame to determine normalization factor
        first_frame = frames_list[0]
        
        if first_frame.dtype == np.uint8:
            # Standard 8-bit video (0-255) -> Divide by 255.0
            norm_factor = 255.0
        elif first_frame.dtype == np.uint16:
            # High bit-depth video (0-65535) -> Divide by 65535.0
            # This handles ProRes/DNxHD correctly without data loss or clipping.
            norm_factor = 65535.0
        else:
            # Fallback for unknown types (rare)
            norm_factor = 255.0

        # Convert frames to float32 and normalize to [0, 1]
        normalized_frames = [f.astype(np.float32) / norm_factor for f in frames_list]
        
        return normalized_frames

    except Exception as e:
        # Debug print can be enabled if specific files fail
        # print(f"Error reading {video_path}: {e}")
        return []

def find_wm_info_path(video_filename, wm_info_root_dir):
    """
    Finds the corresponding wm_info.bin for a video by matching filenames 
    to folder prefixes in the watermark info directory.
    """
    base_name = os.path.splitext(video_filename)[0]
    if not os.path.exists(wm_info_root_dir):
        return None
    
    candidate_folders = [d for d in os.listdir(wm_info_root_dir) if os.path.isdir(os.path.join(wm_info_root_dir, d))]
    best_match = None
    max_len = 0
    
    for folder in candidate_folders:
        if base_name.startswith(folder):
            if len(folder) > max_len:
                max_len = len(folder)
                best_match = folder
                
    if best_match:
         return os.path.join(wm_info_root_dir, best_match, 'wm_info.bin')
    return None

def main(args):
    device = args.device
    model_name = args.model_name
    output_csv = args.output_csv
    
    # --- 1. Load Model ---
    print(f"Loading model: {model_name}...")
    if args.model_path is not None:
        model_path = args.model_path
    else:
        if model_name == 'modelscope':
            # Update this path to your local model directory
            model_path = "/home/david/videomarking/VideoShield-master/Modelscope/"
        elif model_name == 'stable-video-diffusion':
            model_path = 'stabilityai/stable-video-diffusion-img2vid-xt'
        else:
            raise ValueError("Unknown model name")

    if model_name == 'modelscope':
        video_pipe = TextToVideoSDPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    elif model_name == 'stable-video-diffusion':
        video_pipe = StableVideoDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    
    inverse_scheduler = DDIMInverseScheduler.from_pretrained(model_path, subfolder='scheduler')
    video_pipe.scheduler = inverse_scheduler

    attacked_dir = args.attacked_video_dir
    wm_info_dir = args.wm_info_dir
    
    # --- 2. Scan and Group Videos ---
    video_folders = {} 
    # Added .webm and .mkv for VP9/AV1 support
    supported_extensions = ('.mp4', '.avi', '.mov', '.webm', '.mkv')
    
    print(f"Scanning for videos in {attacked_dir}...")
    for root, dirs, files in os.walk(attacked_dir):
        valid_files = [f for f in files if f.lower().endswith(supported_extensions)]
        if valid_files:
            video_folders[root] = [os.path.join(root, f) for f in valid_files]

    sorted_folders = sorted(video_folders.items(), key=lambda x: x[0])
    print(f"Found {len(sorted_folders)} folders containing videos.\n")

    summary_results = {}

    # --- 3. Evaluate Each Folder ---
    for folder_path, video_paths in sorted_folders:
        # Parse path to determine Group Name and Level Name
        rel_path = os.path.relpath(folder_path, attacked_dir)
        path_parts = rel_path.split(os.sep)
        
        # Intelligent naming logic based on folder structure
        if len(path_parts) >= 2:
            group_name = path_parts[0]  # e.g., h264, vp9, prores
            level_name = path_parts[-1] # e.g., Level1_Light, Level5_Extreme
        elif len(path_parts) == 1:
            group_name = path_parts[0]
            level_name = "Default"
        else:
            group_name = "Unknown"
            level_name = os.path.basename(folder_path)

        print(f"--- Processing: Group [{group_name}] - Level [{level_name}] ({len(video_paths)} videos) ---")
        
        folder_total_acc = 0.0
        folder_count = 0
        
        # [Debug Settings] Print errors for the first 3 failed videos per folder
        debug_counter = 0 
        debug_limit = 3

        for video_path in tqdm(video_paths, desc=f"Eval {group_name}/{level_name}"):
            video_filename = os.path.basename(video_path)
            
            # [Step 1] Find Watermark Info
            wm_info_path = find_wm_info_path(video_filename, wm_info_dir)
            if not wm_info_path: 
                if debug_counter < debug_limit:
                    print(f"\n[DEBUG ERROR] wm_info not found for: {video_filename}")
                    debug_counter += 1
                continue

            # Initialize VideoShield
            watermark_module = VideoShield(
                ch_factor=args.channel_copy, 
                hw_factor=args.hw_copy, 
                frame_factor=args.frames_copy,
                height=int(args.height / 8), 
                width=int(args.width / 8), 
                num_frames=args.num_frames, 
                device=device
            )
            
            try:
                wm_info = torch.load(wm_info_path, map_location=device)
                watermark_module.m = wm_info['m']
                watermark_module.watermark = wm_info['watermark']
                watermark_module.key = wm_info['key']
                watermark_module.nonce = wm_info['nonce']
            except Exception as e:
                print(f"[DEBUG ERROR] Error loading wm_info: {e}")
                continue

            # [Step 2] Load Video Frames (Using the new robust function)
            video_frames = load_video_frames(video_path)
            if len(video_frames) == 0:
                if debug_counter < debug_limit:
                    print(f"\n[DEBUG ERROR] Video load failed (0 frames): {video_filename}")
                    print("  -> Possible causes: File corruption, 0KB file, or unsupported codec.")
                    debug_counter += 1
                continue
                
            # [Step 3] Auto-Padding / Frame Count Handling
            # FFmpeg transcoding sometimes drops 1 frame. This logic fixes it.
            target_frames = args.num_frames
            
            if len(video_frames) > target_frames:
                # Case A: Too many frames, trim the excess
                video_frames = video_frames[:target_frames]
            elif len(video_frames) < target_frames:
                # Case B: Too few frames, pad by duplicating the last frame
                # This ensures the video is not skipped and accuracy is calculated.
                while len(video_frames) < target_frames:
                    video_frames.append(video_frames[-1])
            
            # Final Safety Check
            if len(video_frames) != target_frames:
                if debug_counter < debug_limit:
                    print(f"\n[DEBUG ERROR] Frame mismatch impossible: {len(video_frames)} vs {target_frames}")
                    debug_counter += 1
                continue

            # [Step 4] Inversion and Watermark Extraction
            first_frame = cv2_to_pil(video_frames[0])
            video_frames_tensor = transform_video(video_frames).to(video_pipe.vae.dtype).to(device)
            
            if model_name == 'stable-video-diffusion':
                video_latents = get_video_latents(video_pipe.vae, video_frames_tensor, sample=False, permute=False)
            else:
                video_latents = get_video_latents(video_pipe.vae, video_frames_tensor, sample=False, permute=True)

            if model_name == 'modelscope':
                reversed_latents = video_pipe(
                    prompt='',
                    latents=video_latents,
                    num_inference_steps=args.num_inversion_steps,
                    guidance_scale=1.0,
                    output_type='latent',
                ).frames
            elif model_name == 'stable-video-diffusion':
                reversed_latents = video_pipe(
                    image=first_frame,
                    height=args.height,
                    width=args.width,
                    latents=video_latents,
                    num_frames=args.num_frames,
                    output_type='latent',
                    num_inference_steps=args.num_inversion_steps,
                    max_guidance_scale=1.0,
                ).frames
                reversed_latents = reversed_latents.permute(0, 2, 1, 3, 4)
                
            acc = watermark_module.eval_watermark(reversed_latents)
            folder_total_acc += acc
            folder_count += 1

        # Store results for this folder
        if folder_count > 0:
            avg_acc = folder_total_acc / folder_count
            
            if group_name not in summary_results:
                summary_results[group_name] = {}
            summary_results[group_name][level_name] = avg_acc
            
            print(f"  -> Avg Acc: {avg_acc:.4f}\n")
        else:
            print(f"  -> No valid videos processed in this folder.\n")

    # --- 4. Write Detailed CSV ---
    print(f"Writing aggregated results to {output_csv}...")
    
    # Collect all unique Level names for table headers
    all_levels = set()
    for levels_dict in summary_results.values():
        for lvl in levels_dict.keys():
            all_levels.add(lvl)
    sorted_levels = sorted(list(all_levels))
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write Header
        header = ['Group Name'] + sorted_levels + ['Group Average']
        csv_writer.writerow(header)
        
        # Write Rows
        for group_name in sorted(summary_results.keys()):
            row = [group_name]
            level_accs = []
            
            for lvl in sorted_levels:
                acc = summary_results[group_name].get(lvl, None)
                if acc is not None:
                    row.append(f"{acc:.4f}")
                    level_accs.append(acc)
                else:
                    row.append("N/A")
            
            # Calculate overall average for the group
            if level_accs:
                overall_avg = sum(level_accs) / len(level_accs)
                row.append(f"{overall_avg:.4f}")
            else:
                row.append("0.0000")
                
            csv_writer.writerow(row)

    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--model_name', default='modelscope', type=str)
    parser.add_argument('--model_path', default=None, type=str)
    
    parser.add_argument('--attacked_video_dir', required=True, type=str)
    parser.add_argument('--wm_info_dir', required=True, type=str)
    parser.add_argument('--output_csv', default='detailed_eval_results.csv', type=str, help='Path to save the evaluation results csv')
    
    parser.add_argument('--height', default=256, type=int) 
    parser.add_argument('--width', default=256, type=int)
    parser.add_argument('--num_frames', default=16, type=int)
    parser.add_argument('--num_inversion_steps', default=25, type=int)
    
    parser.add_argument('--channel_copy', default=1, type=int)
    parser.add_argument('--frames_copy', default=8, type=int)
    parser.add_argument('--hw_copy', default=4, type=int)

    args = parser.parse_args()
    main(args)