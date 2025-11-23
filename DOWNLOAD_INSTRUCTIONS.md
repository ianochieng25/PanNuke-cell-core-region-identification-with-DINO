# Project Download & Setup Instructions

To run this project, you need to download several large files and datasets that are not included in the GitHub repository.

## 1. PanNuke Dataset

**Description**: The main dataset for nuclei segmentation.
**Download Link**:
- [PanNuke Dataset Part 1](https://www.kaggle.com/datasets/andrewmvd/cancer-inst-segmentation-and-classification)
- [PanNuke Dataset Part 2](https://www.kaggle.com/datasets/andrewmvd/cancer-instance-segmentation-and-classification-2)
- [PanNuke Dataset Part 3](https://www.kaggle.com/datasets/andrewmvd/cancer-instance-segmentation-and-classification-3)

**Action**:
1. Download all 3 parts.
2. Extract the contents. You should find folders corresponding to Fold 1, Fold 2, and Fold 3 across these downloads.
3. Organize them into the `data/pannuke` directory.

**Expected Structure**:
```
pannuke_project/
└── data/
    └── pannuke/
        ├── Fold 1/
        │   ├── images.npy
        │   ├── masks.npy
        │   └── types.npy
        ├── Fold 2/
        │   └── ...
        └── Fold 3/
            └── ...
```

## 2. DINO Pretrained Weights (Optional but Recommended)

**Description**: Official ViT-Small weights pretrained with DINO.
**Download Link**: [dino_deitsmall16_pretrain_full_checkpoint.pth](https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_full_checkpoint.pth)
**Action**:
1. Download the file.
2. Place it in `checkpoints/dino/`.

**Expected Structure**:
```
pannuke_project/
└── checkpoints/
    └── dino/
        └── dino_deitsmall16_pretrain_full_checkpoint.pth
```

## 3. Trained Model Weights (For Inference)

**Description**: If you have trained the model yourself or have a provided checkpoint.
**Action**:
1. Place your best model checkpoint (e.g., `model_final.pth`) in `checkpoints/segmentor/` or the `pannuke_inference_dist/` folder.

**Expected Structure**:
```
pannuke_project/
└── checkpoints/
    └── segmentor/
        └── model_final.pth
```
