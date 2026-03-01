# SAFE-Diff: Structurally Anchored Diffusion for Anatomically Faithful CT Image Super-Resolution

Official PyTorch implementation of **SAFE-Diff**, a two-stage deep learning pipeline for super-resolution enhancement of low-resolution CT scan images.

##  1. Installation & Setup

### Requirements
* **Python:** 3.8+
* **CUDA:** 11.0+ (GPU highly recommended)
* **Framework:** PyTorch 1.12+

### Setup

**Clone the repository**
git clone [https://github.com/nature0012300/SAFE-Diff.git]

**Install core dependencies**
pip install torch torchvision torchaudio numpy pandas pillow scikit-image matplotlib lpips pytorch-fid pywt tqdm

## 2. Pretrained Models & Inference Data
To reproduce our results or run inference on your own samples, you must download the trained weights and the pre-processed dataset indices.

| Resource | Description | Download Link |
| :--- | :--- | :--- |
| **Stage 1 Weights** | Residual Prediction Network  | [**Download from Drive**](https://drive.google.com/drive/folders/1u6-fXL2NzYhwzQNKZdF1fkxE8thxD_pB?usp=sharing) |
| **Stage 2 Weights** | Diffusion Refinement Model | [**Download from Drive**](https://drive.google.com/drive/folders/1u6-fXL2NzYhwzQNKZdF1fkxE8thxD_pB?usp=sharing) |
| **Testing Data** | Normalized CT Patches (LITS/KITS) & Metadata | [**Download from Drive**](https://drive.google.com/drive/folders/1u6-fXL2NzYhwzQNKZdF1fkxE8thxD_pB?usp=sharing) |

### Data Preparation
The system utilizes a CSV-based loading mechanism. You must provide a `.csv` file containing the absolute paths to each CT scan slice.

**CSV Structure Example:**
The CSV file should have a column named `image_path` as shown below:

| image_path |
C
| `/DATA/lits/test_lits/patient_31/slice_90.png` |
| `/DATA/lits/test_lits/patient_128/slice_100.png` |

---

##  3. Configuration & Inference

**Path Configuration**
Before running the evaluation, you **must** update the local directory paths within `eval_main.py` to point to your specific environment for either **LITS** or **KITS** inference.

1. Open `eval_main.py`.
2. Locate the configuration section and update the following variables:


**Edit these paths in eval_main.py**
| :--- |
|'model_path': 'path/to/trained/diffusion/refiner.pth'|
|'stage1_model_path': 'path/to/trained/Stage_1.pth'|
|'test_csv_path': "path/to/your/metadata.csv"|
|'evaluation_output_dir' = "./results/inference_outputs/"|
---


