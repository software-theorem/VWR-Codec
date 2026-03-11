import os
import glob
import subprocess


input_folder = "/home/david/videomarking/videoseal-main/assets/videos/"       
output_folder = "outputs/all_videos" 


os.makedirs(output_folder, exist_ok=True)


videos = glob.glob(os.path.join(input_folder, "*.mp4"))


print(f"find {len(videos)} total")


for video_path in videos:
    print(f"deal: {video_path}")

    cmd = [
        "python", "inference_streaming.py",
        "--input", video_path,
        "--output_dir", output_folder
    ]
    subprocess.run(cmd)

print("finish all")
