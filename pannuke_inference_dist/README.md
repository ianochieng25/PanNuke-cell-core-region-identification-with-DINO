# PanNuke Segmentation Inference

This is a standalone inference package for PanNuke Nuclei Segmentation using a TransUNet model pre-trained with DINO.

## 📂 Structure

*   `inference.py`: Main script for running predictions.
*   `model.py`: TransUNet model definition.
*   `dataset.py`: Data loading utilities.
*   `requirements.txt`: Python dependencies.

## 🛠️ Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Prepare Weights**:
    *   Place your trained model weights (e.g., `model_final.pth`) in this folder or a known location.

## 🚀 Usage

### 1. Run on a Folder of Images (PNG/JPG)

If you have a folder of standard images:

```bash
python inference.py --input "path/to/images" --output "results" --weights "model_final.pth"
```

### 2. Run on PanNuke Dataset (NPY format)

If you are using the original PanNuke dataset structure (with `*_images.npy` files):

```bash
python inference.py --input "path/to/pannuke_data" --output "results" --weights "model_final.pth" --pannuke_format
```

## ⚙️ Arguments

*   `--input`: Path to input directory.
*   `--output`: Path to save results (masks and overlays).
*   `--weights`: Path to the `.pth` model file.
*   `--batch_size`: Batch size (default: 4).
*   `--device`: `cuda` or `cpu` (default: auto-detect).
*   `--alpha`: Overlay transparency (default: 0.4).
*   `--pannuke_format`: Flag to indicate input is PanNuke `.npy` format.
