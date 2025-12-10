# CMPE492 - Multimodal Medical Imaging

Multimodal disease classification using chest X-ray images and clinical text.

## Project Overview

This project implements a multimodal deep learning model for medical image analysis that combines:
- **Vision Model**: SigLIP for chest X-ray image analysis
- **Text Model**: BiomedBERT for clinical indication text processing
- **Fusion Model**: Gated fusion mechanism combining both modalities

The model predicts 14 diseases from the Indiana University chest X-ray dataset.

## Project Structure

```
├── main.py                      # Training script
├── evaluate.py                  # Model evaluation
├── models.py                    # Model architectures
├── data_loaders.py             # Dataset and data loading
├── train_engine.py             # Training and validation loops
├── demo.py                     # Inference/demo script
├── colab_setup.py              # Setup verification for Colab
├── test_vision_loading.py      # Image loading diagnostics
├── diagnose_model.py           # Model diagnostics
├── calculate_class_weights.py  # Calculate class weights
├── check_data_distribution.py  # Analyze dataset distribution
└── verify_fixes.py             # Verify data leakage fixes
```

## Requirements

```
torch>=2.0.0
transformers>=4.30.0
torchvision>=0.15.0
pillow>=9.0.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
```


## Dataset

This project uses the Indiana University Chest X-ray dataset:
- 3,818 chest X-ray images
- Clinical reports with indications, findings, and impressions
- 14 disease labels

**Data structure:**
```
archive/
├── indiana_reports.csv
├── indiana_projections.csv
└── images/
    └── images_normalized/
└── preprocessed_images
```

## Usage

### Training

**Multimodal training (default):**
```bash
python main.py \
  --mode multimodal \
  --epochs 12 \
  --batch 16 \
  --lr 2e-5 \
  --loss weighted_bce \
  --amp
```

**Vision-only training:**
```bash
python main.py \
  --mode vision_only \
  --epochs 12 \
  --batch 32 \
  --lr 1e-4
```

**Text-only training:**
```bash
python main.py \
  --mode text_only \
  --epochs 12 \
  --batch 32 \
  --lr 2e-5
```

### Evaluation

```bash
python evaluate.py \
  --checkpoint checkpoints/multimodal/best_model_best_auc.pth \
  --mode multimodal \
  --split test
```

### Demo/Inference

```bash
python demo.py \
  --multimodal_checkpoint checkpoints/multimodal/best_model_best_auc.pth \
  --mode multimodal
```

## Model Architecture

### Multimodal Model

1. **Vision Encoder**: SigLIP (google/siglip-base-patch16-512)
   - Pretrained on image-text pairs
   - Fine-tuned for X-ray analysis

2. **Text Encoder**: BiomedBERT (microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext)
   - Pretrained on biomedical literature
   - Processes clinical indications

3. **Fusion Mechanism**: Gated Fusion
   - Learns to weight vision vs text contributions
   - Outputs combined disease predictions