Evaluating Video Watermark Robustness Against Codec Attacks (VWR-Codec)
This project provides a comprehensive framework for evaluating the robustness of various video watermarking algorithms against common video codec-based attacks (e.g., H.264/AVC, H.265/HEVC compression).

📂 Repository Structure
The project is organized into three main modules:

algorithms/: Contains the source code for various video/image watermarking methods.

GAN/: Generative Adversarial Network based watermarking.

Video-Signature-main/: Implementation of video signature techniques.

VideoMark-main/: Core video watermarking framework.

VideoShield/: Robust video watermarking modules.

videoseal/: Meta's VideoSeal or similar modern watermarking implementations.

codec_attack/: Tools and scripts to simulate codec-based attacks.

run_all.sh: Entry point to execute batch codec attacks.

universal_worker_new.sh: Worker script for processing individual video files.

evaluation/: Scripts for calculating metrics (e.g., PSNR, SSIM, BER, Bit Accuracy).

quality_check.py: Main script to evaluate visual quality and watermark extraction accuracy.

🚀 Getting Started
1. Installation
Clone the repository and install the required dependencies:

Bash

git clone https://github.com/software-theorem/VWR-Codec.git
cd VWR-Codec
# Each module may have its own requirements
pip install -r algorithms/VideoMark-main/requirements.txt
pip install -r codec_attack/requirements.txt
2. Running an Attack
To test the robustness of watermarked videos against codec attacks, use the provided shell scripts:

Bash

cd codec_attack
bash run_all.sh --input_dir /path/to/videos --output_dir /path/to/results
3. Evaluation
After the attack, evaluate the performance of the watermark:

Bash

python evaluation/quality_check.py --original /path/to/original --attacked /path/to/attacked
📊 Supported Algorithms
Currently, this framework supports benchmarking the following:

VideoSeal (Temporal & Spatial robustness)

VideoMark

Video-Signature

GAN-based Watermarking
