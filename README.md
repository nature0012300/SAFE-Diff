# SAFE-Diff
# SAFE-Diff: Two-Stage Medical Image Super-Resolution

Official PyTorch implementation of **SAFE-Diff**, a two-stage deep learning pipeline for super-resolution enhancement of low-resolution CT scan images, with specific applications in liver and kidney imaging (LITS/KITS).

---

## 🚀 1. Installation & Setup

### Requirements
* **Python:** 3.8+
* **CUDA:** 11.0+ (GPU highly recommended)
* **Framework:** PyTorch 1.12+

### Setup
```bash
# Clone the repository
git clone [https://github.com/nature0012300/SAFE-Diff.git](https://github.com/nature0012300/SAFE-Diff.git)
cd SAFE-Diff

# Install core dependencies
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install numpy pandas pillow scikit-image matplotlib lpips pytorch-fid pywt tqdm
