import os
import argparse
import numpy as np
import torch
import cv2
import pickle
import csv
import imageio  
from tqdm import tqdm
from diffusers import I2VGenXLPipeline, TextToVideoSDPipeline, DDIMInverseScheduler
from utils import transform_video, get_video_latents, cv2_to_pil
from src.prc import Detect, Decode
import src.pseudogaussians as prc_gaussians

# Set random seed
np.random.seed(11111)

def load_video_frames(video_path, target_frames=None):
    """
    Reads video frames using ImageIO (ffmpeg plugin).
    """
    try:
        # 1. Check if file is empty
        if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
            print(f"DEBUG: File is empty or missing: {video_path}")
            return []
            
        reader = imageio.get_reader(video_path, 'ffmpeg')
        frames_list = []
        for frame in reader:
            frames_list.append(frame)
        reader.close()
        
        if len(frames_list) == 0:
            print(f"DEBUG: No frames read from: {video_path}")
            return []

        if target_frames is not None:
            if len(frames_list) > target_frames:
                frames_list = frames_list[:target_frames]
            elif len(frames_list) < target_frames:
                while len(frames_list) < target_frames:
                    frames_list.append(frames_list[-1])
        
        final_frames = []
        for f in frames_list:
            if f.dtype == np.uint16:
                f = (f / 256).astype(np.uint8)
            elif f.dtype != np.uint8:
                f = f.astype(np.uint8)
            final_frames.append(f)

        return final_frames

    except Exception as e:
        print(f"DEBUG: Error reading video {video_path}: {e}")
        return []

def process_frame_extraction(frame_index, reversed_latents, decoding_key):
    latents_vector = reversed_latents[:, :, frame_index].to(torch.float64).flatten()
    reversed_prc = prc_gaussians.recover_posteriors(latents_vector, variances=1.5).flatten()
    return Decode(decoding_key, reversed_prc)

def evaluate_single_video(video_path, pipeline, decoding_key, message_pool, args, debug_counter):
    """
    Process a single video with FIXED matching logic for 'name_0_wm' -> 'name_0'
    """
    if not os.path.exists(video_path):
        return None

    # --- 1. Filename Parsing ---
    filename = os.path.basename(video_path)
    name_without_ext = os.path.splitext(filename)[0]
    
    # Generate potential folder names to search for
    potential_names = []
    
    # Strategy 1: Remove only "_wm" suffix (Target: "video_0")
    if name_without_ext.endswith("_wm"):
        potential_names.append(name_without_ext[:-3]) 
        
    # Strategy 2: Remove "_0_wm" suffix (Target: "video") - For backward compatibility
    if name_without_ext.endswith("_0_wm"):
        potential_names.append(name_without_ext[:-5])

    # Strategy 3: Exact match (Just in case)
    potential_names.append(name_without_ext)

    shift_path = None
    found_name = None

    # Try to find which folder actually exists in original_dir
    for name in potential_names:
        temp_path = os.path.join(args.original_dir, name, "shift_value.npy")
        if os.path.exists(temp_path):
            shift_path = temp_path
            found_name = name
            break
    
    # --- DEBUGGING OUTPUT ---
    if shift_path is None:
        if debug_counter[0] < 3: # Limit errors to 3 lines
            print(f"\n[MATCH FAILED] for: {filename}")
            print(f"   - We tried these folder names: {potential_names}")
            print(f"   - In directory: {args.original_dir}")
            debug_counter[0] += 1
        return None

    try:
        # --- 2. Load Ground Truth ---
        shift = np.load(shift_path)
        ground_truth_bits = message_pool[shift : shift + args.num_frames]

        # --- 3. Read Video ---
        frames = load_video_frames(video_path, target_frames=args.num_frames)
        
        if len(frames) != args.num_frames:
            # Silent fail for frame mismatch to keep logs clean, unless critical
            return None 

        # --- 4. Watermark Extraction ---
        video_tensor = transform_video(frames).to(pipeline.vae.dtype).to(args.device)

        with torch.no_grad():
            video_latents = get_video_latents(pipeline.vae, video_tensor, sample=False, permute=True)
            
            if args.model_name == 'modelscope':
                reversed_latents = pipeline(
                    prompt='',
                    latents=video_latents,
                    num_inference_steps=50,
                    guidance_scale=1.0,
                    output_type='latent',
                ).frames
            elif args.model_name == 'i2vgen-xl':
                reversed_latents = pipeline(
                    prompt="None",
                    image=cv2_to_pil(frames[0]),
                    height=512, width=512,
                    latents=video_latents.half(),
                    num_frames=args.num_frames,
                    output_type='latent',
                    num_inference_steps=50,
                    guidance_scale=9,
                ).frames

        # --- 5. Calculate Accuracy ---
        reversed_latents_cpu = reversed_latents.cpu()
        total_bits = 0
        correct_bits = 0

        for i in range(args.num_frames):
            extracted_bits = process_frame_extraction(i, reversed_latents_cpu, decoding_key)
            gt_bits_frame = ground_truth_bits[i]

            if len(extracted_bits) == len(gt_bits_frame):
                matches = np.sum(extracted_bits == gt_bits_frame)
                total_bits += len(gt_bits_frame)
                correct_bits += matches
        
        if total_bits == 0: return 0.0
        return correct_bits / total_bits

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def main(args):
    print(f"Loading Model: {args.model_name}...")
    if args.model_name == 'modelscope':
        model_path = "/home/david/videomarking/VideoMark-main/hf_models/modelscope/"
        pipeline = TextToVideoSDPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to(args.device)
    elif args.model_name == 'i2vgen-xl':
        model_path = 'ali-vilab/i2vgen-xl' 
        pipeline = I2VGenXLPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to(args.device)
    else:
        raise ValueError("Unsupported model.")

    pipeline.scheduler = DDIMInverseScheduler.from_pretrained(model_path, subfolder='scheduler')

    keys_file = os.path.join(args.keys_path, f"{args.height}_{args.width}_{args.num_bit}bit.pkl")
    if not os.path.exists(keys_file):
        print(f"CRITICAL ERROR: Keys file missing at {keys_file}")
        return

    with open(keys_file, 'rb') as f:
        _, decoding_key = pickle.load(f)

    message_bits_sequence = np.random.randint(0, 2, size=(500, args.num_bit))

    print(f"Original Dir Check: {args.original_dir}")
    if os.path.exists(args.original_dir):
        print(f"First 5 items in Original Dir: {os.listdir(args.original_dir)[:5]}")
    else:
        print("CRITICAL ERROR: Original Directory does not exist!")
        return

    csv_file = open(args.output_csv, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Codec", "Level", "Video_Count", "Average_Bit_Accuracy"])

    valid_extensions = ('.mp4', '.mov', '.mkv', '.webm', '.avi')

    def process_files(file_list, current_dir, codec_name, level_name):
        accuracies = []
        debug_counter = [0] # List to pass by reference for counting errors
        
        for vid_file in tqdm(file_list, desc=f"{codec_name}-{level_name}"):
            video_full_path = os.path.join(current_dir, vid_file)
            acc = evaluate_single_video(
                video_full_path, 
                pipeline, 
                decoding_key, 
                message_bits_sequence, 
                args,
                debug_counter
            )
            if acc is not None:
                accuracies.append(acc)
        
        if accuracies:
            avg_acc = sum(accuracies) / len(accuracies)
            print(f"? Result for {codec_name}/{level_name}: {avg_acc:.4f}")
            csv_writer.writerow([codec_name, level_name, len(accuracies), f"{avg_acc:.4f}"])
            csv_file.flush()
        else:
            print(f"? No valid results for {codec_name}/{level_name}")

    root_files = [f for f in os.listdir(args.root_dir) if f.lower().endswith(valid_extensions)]
    
    if len(root_files) > 0:
        print("Detected video files in root directory. Processing as single batch...")
        path_parts = os.path.normpath(args.root_dir).split(os.sep)
        level_name = path_parts[-1]
        codec_name = path_parts[-2] if len(path_parts) > 1 else "Unknown"
        process_files(root_files, args.root_dir, codec_name, level_name)
    else:
        codecs = sorted([d for d in os.listdir(args.root_dir) if os.path.isdir(os.path.join(args.root_dir, d))])
        for codec in codecs:
            codec_path = os.path.join(args.root_dir, codec)
            levels = sorted([d for d in os.listdir(codec_path) if os.path.isdir(os.path.join(codec_path, d))])
            for level in levels:
                level_path = os.path.join(codec_path, level)
                video_files = [f for f in os.listdir(level_path) if f.lower().endswith(valid_extensions)]
                if not video_files: continue
                print(f"\n>>> Processing Group: {codec} / {level} | Found {len(video_files)} videos")
                process_files(video_files, level_path, codec, level)

    csv_file.close()
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--original_dir', type=str, required=True)
    parser.add_argument('--keys_path', default="./keys", type=str)
    parser.add_argument('--output_csv', default="attack_results.csv", type=str)
    parser.add_argument('--model_name', default='modelscope', type=str)
    parser.add_argument('--num_bit', default=512, type=int)
    parser.add_argument('--height', default=64, type=int)
    parser.add_argument('--width', default=64, type=int)
    parser.add_argument('--num_frames', default=16, type=int)
    parser.add_argument('--device', default='cuda:0')
    
    args = parser.parse_args()
    main(args)