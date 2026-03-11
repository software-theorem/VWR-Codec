import cv2
import numpy as np
import torch
import lpips
import os
import datetime
import imageio.v2 as imageio  # Use ImageIO v2 API for stability
from skimage.metrics import structural_similarity as ssim_func

class VideoQualityEvaluator:
    def __init__(self, use_gpu=True):
        """
        Initialize the evaluator, set up the device (GPU/CPU), and load the LPIPS model.
        """
        # 1. Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        print(f"Using device: {self.device}")

        # 2. Initialize LPIPS model
        # 'net=alex' is standard for LPIPS.
        self.lpips_model = lpips.LPIPS(net='alex', verbose=False).to(self.device)

    def preprocess_for_lpips(self, frame):
        """
        Preprocess a frame for LPIPS calculation.
        Note: ImageIO reads frames in RGB, so we do NOT need BGR->RGB conversion.
        """
        # Normalize to [-1, 1] range
        img = (frame / 255.0) * 2 - 1
        # Convert to Tensor: (H, W, C) -> (C, H, W) -> Batch (1, C, H, W)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        return img_tensor.to(self.device)

    def calculate_metrics(self, video_path_ref, video_path_dist, output_txt, sample_interval=1):
        """
        Calculate PSNR, SSIM, and LPIPS for a pair of videos using ImageIO.
        """
        if not os.path.exists(video_path_ref) or not os.path.exists(video_path_dist):
            return None

        reader_ref = None
        reader_dist = None

        try:
            # Open videos using ImageIO with the 'ffmpeg' plugin (better AV1/H.265 support)
            reader_ref = imageio.get_reader(video_path_ref, 'ffmpeg')
            reader_dist = imageio.get_reader(video_path_dist, 'ffmpeg')
            
            # Get metadata to check dimensions
            meta_ref = reader_ref.get_meta_data()
            w_ref, h_ref = meta_ref['size'] # imageio uses (width, height)
            
        except Exception as e:
            # This catches "moov atom not found" or "invalid data" errors
            print(f"  [Error] Could not open video (File might be corrupt): {os.path.basename(video_path_dist)}")
            print(f"  [Detail] {e}")
            return None

        # Logging context for the user
        try:
            prompt_name = os.path.basename(os.path.dirname(os.path.dirname(video_path_ref)))
        except:
            prompt_name = os.path.basename(os.path.dirname(video_path_ref))
        dist_name = os.path.basename(video_path_dist)
        ref_name = os.path.basename(video_path_ref)

        print(f"  [Processing] Ref: {ref_name} | Dist: {dist_name}")

        psnr_list, ssim_list, lpips_list = [], [], []

        try:
            # Iterate over both videos simultaneously
            # enumerate is used to handle the sample_interval
            for frame_idx, (frame_ref, frame_dist) in enumerate(zip(reader_ref, reader_dist)):
                
                # Skip frames based on interval to save time
                if frame_idx % sample_interval != 0:
                    continue

                # --- Resize Logic ---
                # ImageIO returns frames as numpy arrays (Height, Width, Channel)
                h_d, w_d = frame_dist.shape[:2]
                
                # If dimensions don't match, resize the distorted frame to match reference
                if (h_ref != h_d) or (w_ref != w_d):
                    # cv2.resize expects (Width, Height)
                    frame_dist = cv2.resize(frame_dist, (w_ref, h_ref), interpolation=cv2.INTER_LINEAR)

                # --- Metrics Calculation ---
                
                # 1. PSNR (OpenCV's PSNR works fine with RGB as long as both inputs are RGB)
                psnr_val = cv2.PSNR(frame_ref, frame_dist)
                
                # 2. SSIM (skimage supports multi-channel images)
                ssim_val = ssim_func(frame_ref, frame_dist, data_range=255, channel_axis=2)
                
                # 3. LPIPS (Requires PyTorch tensors)
                t_ref = self.preprocess_for_lpips(frame_ref)
                t_dist = self.preprocess_for_lpips(frame_dist)
                
                with torch.no_grad():
                    lpips_val = self.lpips_model(t_ref, t_dist).item()

                psnr_list.append(psnr_val)
                ssim_list.append(ssim_val)
                lpips_list.append(lpips_val)

        except Exception as e:
            print(f"  [Warning] Error occurred during frame processing: {e}")
            pass
            
        finally:
            # Ensure resources are released
            if reader_ref: reader_ref.close()
            if reader_dist: reader_dist.close()

        # If no frames were processed successfully, return None
        if not psnr_list:
            return None

        # Calculate averages
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        avg_lpips = np.mean(lpips_list)

        # Write individual result to the report file
        self.save_results_to_txt(output_txt, video_path_ref, video_path_dist, avg_psnr, avg_ssim, avg_lpips)
        
        return {"PSNR": avg_psnr, "SSIM": avg_ssim, "LPIPS": avg_lpips}

    def save_results_to_txt(self, txt_path, ref_path, dist_path, psnr, ssim, lpips_val):
        """
        Append the results of a single video pair to the text file.
        """
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            # Attempt to shorten the path for cleaner logging
            ref_parts = ref_path.split(os.sep)
            ref_short = f"{ref_parts[-3]}/{ref_parts[-2]}/{ref_parts[-1]}"
        except:
            ref_short = os.path.basename(ref_path)
        dist_name = os.path.basename(dist_path)

        log_content = (
            f"[{current_time}]\n"
            f"Ref : .../{ref_short}\n"
            f"Dist: {dist_name}\n"
            f"Metrics: PSNR={psnr:.4f}, SSIM={ssim:.4f}, LPIPS={lpips_val:.4f}\n"
            f"--------------------------------------------------\n"
        )
        try:
            with open(txt_path, "a", encoding="utf-8") as f:
                f.write(log_content)
        except Exception:
            pass

    def evaluate_folder_recursive(self, ref_root, dist_root, output_txt="batch_report.txt", sample_interval=1):
        """
        Recursively walk through the dist_root, find video files, match them with 
        reference videos in ref_root, and calculate metrics.
        """
        print("="*60)
        print(f"Starting Evaluation")
        print(f"Ref Root: {ref_root}")
        print(f"Dist Root: {dist_root}")
        print("="*60)

        valid_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
        
        # Walk through directory tree
        for root, dirs, files in os.walk(dist_root):
            # Filter for video files
            video_files = [f for f in files if f.lower().endswith(valid_extensions)]
            
            if not video_files:
                continue

            # Identify current subfolder relative to dist_root for cleaner logs
            rel_folder = os.path.relpath(root, dist_root)
            print(f"\n>>> Processing Folder: {rel_folder} ({len(video_files)} videos)")
            
            folder_metrics = {"PSNR": [], "SSIM": [], "LPIPS": []}

            for file in video_files:
                path_dist = os.path.join(root, file)
                
                # --- MATCHING LOGIC FIXED HERE ---
                path_ref = None
                
                candidates = []
                
                # Priority 1: Remove "all_videos_" prefix AND force .mp4 extension
                # (Example: all_videos_VID_2025.webm -> VID_2025.mp4)
                if "all_videos_" in file:
                    temp_name = file.replace("all_videos_", "")
                    base_name = os.path.splitext(temp_name)[0]
                    candidates.append(base_name + ".mp4")

                # Priority 2: Just force .mp4 extension (Example: VID_2025.mov -> VID_2025.mp4)
                base_name_simple = os.path.splitext(file)[0]
                candidates.append(base_name_simple + ".mp4")
                
                # Priority 3: Exact name match
                candidates.append(file)
                
                # Check which candidate exists in Ref Root
                for cand in candidates:
                    check_path = os.path.join(ref_root, cand)
                    if os.path.exists(check_path):
                        path_ref = check_path
                        break
                
                if not path_ref:
                    print(f"  [Skip] Ref not found for: {file}")
                    # print(f"         Tried: {candidates}")
                    continue

                # --- Run Calculation ---
                metrics = self.calculate_metrics(path_ref, path_dist, output_txt, sample_interval)
                
                if metrics:
                    folder_metrics["PSNR"].append(metrics["PSNR"])
                    folder_metrics["SSIM"].append(metrics["SSIM"])
                    folder_metrics["LPIPS"].append(metrics["LPIPS"])

            # === FOLDER SUMMARY ===
            # Calculate average metrics for the current folder
            count = len(folder_metrics["PSNR"])
            if count > 0:
                avg_p = np.mean(folder_metrics["PSNR"])
                avg_s = np.mean(folder_metrics["SSIM"])
                avg_l = np.mean(folder_metrics["LPIPS"])
                
                summary = (
                    f"\n[SUMMARY FOR FOLDER: {rel_folder}]\n"
                    f"Videos Processed: {count}\n"
                    f"AVG PSNR : {avg_p:.4f}\n"
                    f"AVG SSIM : {avg_s:.4f}\n"
                    f"AVG LPIPS: {avg_l:.4f}\n"
                    f"{'='*50}\n"
                )
                print(summary)
                with open(output_txt, "a", encoding="utf-8") as f:
                    f.write(summary)
            else:
                print(f"  [Info] No valid pairs processed in {rel_folder}")

        print("\nALL TASKS COMPLETE")

# --- Main Execution Block ---
if __name__ == "__main__":
    evaluator = VideoQualityEvaluator(use_gpu=True)
    
    # 1. Ref Root (Original Clean/Reference Videos)
    clean_folder = "/home/david/videomarking/videoseal-main/outputs/all_videos/"
    
    # 2. Dist Root (Parent folder containing all h265/LevelX folders)
    target_folder = "/home/david/videomarking/codec/videoseal_time_result/"
    
    # 3. Output Report File
    report_file = "All_Levels_Report.txt"
    
    if os.path.exists(clean_folder) and os.path.exists(target_folder):
        # sample_interval=10 processes 1 out of every 10 frames to speed up execution
        # You can change this to 1 for full accuracy, but it will be slower
        evaluator.evaluate_folder_recursive(clean_folder, target_folder, report_file, sample_interval=10)
    else:
        print("[ERROR] Check your paths in the __main__ block.")