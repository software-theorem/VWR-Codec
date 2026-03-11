import os
import sys
import numpy as np
import torch
import argparse
import warnings
import csv
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import imageio # Make sure to install: pip install imageio[ffmpeg]

warnings.filterwarnings("ignore")

# ================= Configuration =================
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

normalize_transform = transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)

# ================= Core Reading Logic=================
def load_video_frames(video_path):
    try:
        if not os.path.exists(video_path):
            return []
        reader = imageio.get_reader(video_path, 'ffmpeg')
        frames_list = []
        for frame in reader:
            frames_list.append(frame)
        reader.close()
        
        if len(frames_list) == 0:
            return []

        first_frame = frames_list[0]
        if first_frame.dtype == np.uint8:
            norm_factor = 255.0
        elif first_frame.dtype == np.uint16:
            norm_factor = 65535.0
        else:
            norm_factor = 255.0

        normalized_frames = [f.astype(np.float32) / norm_factor for f in frames_list]
        return normalized_frames
    except Exception as e:
        print(f"Error reading {video_path}: {e}")
        return []

def str_to_list(s):
    return [int(char) for char in s]

def list_to_torch(l):
    return torch.tensor(l, dtype=torch.float32)

# ================= Dataset Class================
class AttackedVideoDataset(Dataset):
    def __init__(self, folder_path: str, recursive: bool = False):
        """

        """
        self.files = self._scan_files(folder_path, recursive)
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        filepath = self.files[idx]
        ext = os.path.splitext(filepath)[1].lower()
        frames_list = []
        
        if ext == '.npy':
            frames_np = np.load(filepath, mmap_mode=None)
            if frames_np.dtype == np.uint8:
                frames_np = frames_np.astype(np.float32) / 255.0
            frames_list = [f for f in frames_np] 
        else:
            frames_list = load_video_frames(filepath)
            
        if not frames_list:
            return torch.zeros(1, 3, 256, 256), filepath

        frames_np = np.stack(frames_list)
        video_tensor = torch.from_numpy(frames_np)
        video_tensor = video_tensor.permute(0, 3, 1, 2)
        video_tensor = torch.stack([normalize_transform(f) for f in video_tensor])
        
        return video_tensor, filepath

    def _scan_files(self, path: str, recursive: bool):
        supported_ext = ['.npy', '.mp4', '.avi', '.mkv', '.mov', '.webm']
        files = []
        
        if os.path.isfile(path):
            return [path]
            
        if recursive:
            for current_path, _, filenames in os.walk(path):
                for filename in filenames:
                    if os.path.splitext(filename)[1].lower() in supported_ext:
                        files.append(os.path.join(current_path, filename))
        else:
            for filename in os.listdir(path):
                full_path = os.path.join(path, filename)
                if os.path.isfile(full_path) and os.path.splitext(filename)[1].lower() in supported_ext:
                    files.append(full_path)
                    
        return sorted(files)

# ================= Main Verification Logic=================
def verify(params):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running evaluation on: {device}")

    # 1. Load Decoder
    try:
        msg_decoder = torch.jit.load(params.msg_decoder_path).to(device)
        msg_decoder.eval()
    except Exception as e:
        print(f"Failed to load decoder: {e}")
        return

    # 2. Prepare Key
    key_list = str_to_list(params.key)
    target_key = list_to_torch(key_list).to(device)

    target_folders = []
    supported_ext = ('.npy', '.mp4', '.avi', '.mkv', '.mov', '.webm')
    
    print(f"Scanning directory structure in: {params.video_path} ...")
    for root, dirs, files in os.walk(params.video_path):
        has_video = any(f.lower().endswith(supported_ext) for f in files)
        if has_video:
            target_folders.append(root)
    
    target_folders.sort()
    
    if not target_folders:
        print("No folders containing video files were found!")
        return

    print(f"Found {len(target_folders)} folders to evaluate.\n")

    summary_results = [] # List of [Folder Name, Average Acc]

    # 4. Iterate over each folder
    total_folders = len(target_folders)
    
    for i, folder_path in enumerate(target_folders):
        rel_path = os.path.relpath(folder_path, params.video_path)
        print(f"[{i+1}/{total_folders}] Processing Folder: {rel_path}")
        
        dataset = AttackedVideoDataset(folder_path, recursive=False)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
        
        folder_accuracies = []
        
        with torch.no_grad():
            for frames, filepath in dataloader:
                if frames.dim() == 5: frames = frames.squeeze(0)
                frames = frames.to(device).to(torch.float32)

                decoded = msg_decoder(frames)

                binarized = (decoded > 0).int()
                votes = binarized.sum(dim=0)
                majority_vote = (votes > (decoded.size(0) // 2)).int().reshape(target_key.shape)
                
                matches = (~torch.logical_xor(majority_vote, target_key > 0))
                acc = torch.sum(matches).item() / matches.numel()
                
                folder_accuracies.append(acc)

        if folder_accuracies:
            avg_acc = np.mean(folder_accuracies)
            print(f"  -> Average Acc: {avg_acc:.4f} ({avg_acc*100:.2f}%)")
            print("-" * 40)
            summary_results.append([rel_path, avg_acc])
        else:
            print(f"  -> No valid videos processed.")
            summary_results.append([rel_path, 0.0])

    # 5. Final Summary Output
    print("\n" + "="*50)
    print("FINAL SUMMARY REPORT")
    print("="*50)
    print(f"{'Folder Path':<40} | {'Avg Acc':<10}")
    print("-" * 55)
    
    grand_total_acc = []
    
    if params.csv_file:
        with open(params.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Folder Path', 'Average Accuracy']) # Header
            
            for folder_name, acc in summary_results:
                print(f"{folder_name:<40} | {acc:.4f}")
                writer.writerow([folder_name, f"{acc:.4f}"])
                grand_total_acc.append(acc)
                
    if grand_total_acc:
        print("-" * 55)
        print(f"Global Average (All Folders): {np.mean(grand_total_acc):.4f}")
        print("="*50)
        print(f"Results saved to: {params.csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, required=True, help='Root directory containing subfolders of videos')
    parser.add_argument('--msg_decoder_path', type=str, default='./ckpts/msg_decoder/dec_48b_whit.torchscript.pt')
    parser.add_argument('--key', type=str, default='100011100001001101101100100011111101111110000000')
    parser.add_argument('--csv_file', type=str, default='evaluation_results.csv', help='Output CSV file path')
    
    params = parser.parse_args()
    verify(params)