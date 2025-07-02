# CAVALRY-V: A Large-Scale Generator Framework for Adversarial Attacks on Video MLLMs

[![arXiv](https://img.shields.io/badge/arXiv-2507.00817-b31b1b.svg)](https://arxiv.org/abs/2507.00817)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)

> **Official implementation of "CAVALRY-V: A Large-Scale Generator Framework for Adversarial Attacks on Video MLLMs"**

## 🚀 Overview

CAVALRY-V is a comprehensive framework for generating adversarial attacks against Video Multimodal Large Language Models (MLLMs). Our approach introduces a novel generator-based method that can efficiently produce adversarial perturbations for both image and video inputs, enabling systematic evaluation of video MLLM robustness.

### Key Features

- 🎯 **Universal Generator**: Single framework for both image and video adversarial attacks
- 📊 **Comprehensive Evaluation**: Support for MMBench-Video, MME, and custom datasets
- ⚡ **Efficient Generation**: Open-source large-scale generator
- 🛠️ **Easy Integration**: Compatible with VLMEvalKit for standardized evaluation

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (recommended)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/jiamingzhang94/CAVALRY-V.git
cd CAVALRY-V

# Install dependencies
pip install -r requirements.txt

# Install VLMEvalKit for evaluation
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
pip install -e .
```

## 📊 Data Preparation

### Required Datasets

| Dataset | Purpose | Download Link |
|---------|---------|---------------|
| **MMBench-Video** | Video evaluation | [HuggingFace](https://huggingface.co/datasets/opencompass/MMBench-Video) |
| **MME** | Image evaluation | Auto-downloaded via HuggingFace |

### Optional Datasets (for training from scratch)

| Dataset | Purpose | Download Link |
|---------|---------|---------------|
| **LAION-400M** | Generator pre-training | [WebDataset](https://github.com/webdataset/webdataset) |
| **LLaVA-Instruct-150K** | Visual instruction tuning | Auto-downloaded via HuggingFace |
| **Video-MME** | Video finetuning | Auto-downloaded via HuggingFace |

## 🏃‍♂️ Quick Start

### 1. Download Pre-trained Generator

Download our pre-trained generator weights:
- **Download Link**: [SharePoint](https://entuedu-my.sharepoint.com/:u:/g/personal/jiaming_zhang_staff_main_ntu_edu_sg/EfUgYFlfewVJueEMYOWKz3gBW47d7YtIenwmHhLyVDEjRQ?e=rU1mHR)
- **Password**: `reg&fw12KJ`

### 2. Generate Adversarial Videos

```bash
python generate_video.py \
    --generator_checkpoint /path/to/generator/weights \
    --video_input_dir /path/to/MMBench-Video \
    --video_output_dir /path/to/output/adversarial/videos \
    --qa_json dataset/mmbench-video-qa.json
```

**Note**: Generation on MMBench-Video takes several hours on a single GPU. For faster processing:
- You can run multiple instances of the generation command with different GPU assignments
- Each instance will automatically skip already processed videos (supported via lines 157-183 in `generate_video.py`)
- Remove any `.inprogress` files after completion

### 3. Generate Adversarial Images

```bash
python generate_img.py \
    --generator_checkpoint /path/to/generator/weights \
    --input_dir /path/to/MME \
    --output_dir /path/to/output/adversarial/images \
    --raw_tsv /path/to/MME.tsv \
    --output_tsv /path/to/output/adversarial/MME.tsv
```

Download MME.tsv from: [OpenCompass](https://opencompass.openxlab.space/utils/VLMEval/MME.tsv)

<details>
<summary>💡 <strong>Tips for Image Generation</strong></summary>

- Image generation is much faster than video generation
- `--output_dir` is where results are saved - you can easily modify this script to adapt to your own datasets  
- `--output_tsv` generates the TSV file needed for VLMEval evaluation - this is optional if you don't plan to use VLMEval

</details>

## 📈 Evaluation

### Setup VLMEvalKit

> ⚠️ **Note**: This step can be somewhat complex and requires careful environment management due to dependency conflicts between different models.

1. **Configure API Keys**: Set up your evaluation model API keys following [VLMEvalKit documentation](https://github.com/open-compass/VLMEvalKit)
   
   <details>
   <summary>💡 <strong>Alternative Evaluation Options</strong></summary>
   
   - VLMEvalKit also provides free open-source models as scorers, though they may be slightly less effective than GPT-based evaluators
   
   </details>

2. **Deploy Target Models**: Install the MLLMs you want to evaluate:
   - Qwen2.5-VL-7B-Instruct
   - llava_video_qwen2_7b
   - Aria
   - InternVL2_5-8B
   - MiniCPM-o-2_6

   <details>
   <summary>🔧 <strong>Environment Management Tips</strong></summary>
   
   **Transformer Version Conflicts**: Different models have conflicting transformer library requirements. Based on our experience:
   - Qwen and InternVL can share the same conda environment
   - Other models (llava_video, Aria, MiniCPM-o) maybe should each have their own separate conda environment
   
   </details>

3. **Apply Custom Modifications**:
   ```bash
   # Copy our custom evaluation files to override original VLMEval folder
   cp -r VLMEvalKit/vlmeval/* /path/to/your/VLMEvalKit/vlmeval/
   ```


4. **Update Configuration Paths**:
   - Edit `VLMEvalKit/vlmeval/dataset/video_dataset_config.py:11` → Set your adversarial video output path
   - Edit `VLMEvalKit/vlmeval/dataset/image_yorn.py:13` → Set your adversarial image TSV path

### Run Evaluation

```bash
cd VLMEvalKit
./run.sh
```

<details>
<summary>⚠️ <strong>Important Evaluation Guidelines</strong></summary>

**Best Practices**:
- **One at a time**: Evaluate only one dataset and one model per run to avoid conflicts
- **Cache management**: After each evaluation run, manually delete cache files in:
  - `output_${data}` directory
  - `LMUData` directory
- **Why clean cache**: This prevents the evaluator from reading stale cached data that could cause errors in subsequent evaluations

</details>

## 🏗️ Project Structure

```
CAVALRY-V/
├── dataset/                    # Dataset configurations
│   └── mmbench-video-qa.json
├── VLMEvalKit/                # Custom evaluation scripts
│   ├── run.sh
│   └── vlmeval/
├── attack_utils.py            # Adversarial attack utilities
├── config.py                  # Configuration settings
├── generate_img.py            # Image adversarial generation
├── generate_video.py          # Video adversarial generation
├── model_utils.py             # Model utilities
├── pretrain.py                # Stage 1: Pre-training
├── visual_finetuning.py       # Stage 2: Visual finetuning
├── video_finetuning.py        # Stage 3: Video finetuning
└── requirements.txt           # Dependencies
```

## 🔬 Training from Scratch (Optional)

### Stage 1: Pre-training on LAION-400M

```bash
torchrun --nproc_per_node=8 pretrain.py \
    --image_input_dir /path/to/laion400m/tars \
    --save_path ./weights/laion/
```

### Stage 2: Visual Finetuning

```bash
torchrun --nproc_per_node=8 visual_finetuning.py \
    --generator_checkpoint ./weights/laion/checkpoint_latest.pth \
    --save_path ./weights/visual/
```

### Stage 3: Video Finetuning

```bash
torchrun --nproc_per_node=8 video_finetuning.py \
    --generator_checkpoint ./weights/visual/checkpoint_latest.pth \
    --save_path ./weights/video/
```

## 📝 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{zhang2025cavalry,
  title={CAVALRY-V: A Large-Scale Generator Framework for Adversarial Attacks on Video MLLMs},
  author={Zhang, Jiaming and Hu, Rui and Guo, Qing and Lim, Wei Yang Bryan},
  journal={arXiv preprint arXiv:2507.00817},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔧 Customization and Extensions

<details>
<summary>🛠️ <strong>Adapting the Generator to Your Own Datasets</strong></summary>

**For Custom Image Datasets**:
- The `generate_img.py` script can be easily modified to work with your own datasets
- Simply adjust the input/output paths and modify the image loading logic if needed
- The TSV output format follows VLMEval standards but can be customized for other evaluation frameworks

**For Custom Video Datasets**:
- Modify the video loading and processing pipeline in `generate_video.py`
- Ensure your video format is supported (MP4 recommended)
- Adjust the QA JSON format to match your dataset structure


</details>


## 🙏 Acknowledgments

- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) for evaluation framework
- [MMBench-Video](https://huggingface.co/datasets/opencompass/MMBench-Video) for benchmark dataset
- [LAION-400M](https://laion.ai/) for pre-training data

---

<div align="center">
  <p>⭐ If you find this project helpful, please consider giving it a star! ⭐</p>
</div>