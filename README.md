# Evaluating Video Watermark Robustness Against Codec Attacks (VWR-Codec)

This project provides a comprehensive framework for evaluating the robustness of various video watermarking algorithms against common video codec-based attacks (e.g., H.264/AVC, H.265/HEVC compression).

---

## Repository Structure

The project is organized into three main modules:

### 1. Algorithms
Contains the source code for various video/image watermarking methods:
* **`algorithms/GAN/`**: Generative Adversarial Network based watermarking.
* **`algorithms/Video-Signature-main/`**: Implementation of video signature techniques.
* **`algorithms/VideoMark-main/`**: Core video watermarking framework.
* **`algorithms/VideoShield/`**: Robust video watermarking modules.
* **`algorithms/videoseal/`**: Meta's VideoSeal modern watermarking implementation.

### 2. Codec Attack
Tools and scripts to simulate codec-based attacks:
* **`run_all.sh`**: Entry point to execute batch codec attacks.
* **`universal_worker_new.sh`**: Worker script for processing individual video files.
* **`requirements.txt`**: Dependencies for the attack environment.

### 3. Evaluation
Scripts for calculating metrics and verifying results:
* **`quality_check.py`**: Main script to evaluate visual quality and watermark extraction accuracy.
* **Metrics supported**: PSNR, SSIM, BER, Bit Accuracy.

---

## Getting Started

### 1. Installation
Clone the repository and enter the directory:

```bash
git clone [https://github.com/software-theorem/VWR-Codec.git](https://github.com/software-theorem/VWR-Codec.git)
cd VWR-Codec

---

## ⚙️ 2. Environment Setup

Since different algorithms may have different dependencies, it is recommended to install them sequentially:

* **`codec_attack/requirements.txt`**: Core dependencies for the attack framework.
* **`algorithms/VideoMark-main/requirements.txt`**: Specific dependencies for algorithms like VideoMark.

```bash
# Install core attack dependencies
pip install -r codec_attack/requirements.txt

# Install specific algorithm dependencies (Example: VideoMark)
pip install -r algorithms/VideoMark-main/requirements.txt
