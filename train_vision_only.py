#!/usr/bin/env python3
"""
Train vision-only model for disease classification
Simplified training focused on getting vision to work
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from transformers import AutoImageProcessor
from pathlib import Path
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_loaders import IndianaMultimodalDataset
from models import VisionDiseaseModel

# Dummy tokenizer for dataset compatibility
class DummyTokenizer:
    def __call__(self, text, **kwargs):
        return {
            'input_ids': torch.zeros(256, dtype=torch.long),
            'attention_mask': torch.zeros(256, dtype=torch.long)
        }

def train_vision_model():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Clear GPU memory first
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory cleared")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    
    print(f"Using device: {device}\n")
    
    # Paths
    reports = '/content/drive/MyDrive/archive/indiana_reports.csv'
    projections = '/content/drive/MyDrive/archive/indiana_projections.csv'
    img_dir = '/content/drive/MyDrive/archive/preprocessed_images'
    
    # Hyperparameters
    batch_size = 16  # Reduced to prevent OOM
    epochs = 12
    lr = 1e-4  # Higher LR for vision
    
    print("="*70)
    print("  VISION-ONLY TRAINING")
    print("="*70)
    
    # Load processor
    processor = AutoImageProcessor.from_pretrained('google/siglip-base-patch16-512', use_fast=True)
    tokenizer = DummyTokenizer()  # Dummy for compatibility
    
    # Create dataset
    print("\nLoading dataset...")
    dataset = IndianaMultimodalDataset(
        reports, projections, img_dir,
        processor, tokenizer,
        use_enhancement=False,
        use_bone_suppression=False,
        augment=False  # Start without augmentation
    )
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    
    indices = list(range(total_size))
    random.seed(42)
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=2, pin_memory=True)
    
    print(f"Train samples: {len(train_set)}")
    print(f"Val samples: {len(val_set)}")
    
    # Create model
    print("\nCreating model...")
    model = VisionDiseaseModel('google/siglip-base-patch16-512', freeze_backbone=False).to(device)
    
    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Weighted BCE loss to handle class imbalance
    # Weights calculated from dataset: Effusion (2552 samples) vs Fibrosis (10 samples)
    pos_weights = torch.tensor([
        30.30,   # Pneumonia
        27.92,   # Cardiomegaly
        18.99,   # Edema
        0.50,    # Effusion (very common, low weight)
        16.51,   # Atelectasis
        0.63,    # Pneumothorax (very common, low weight)
        18.38,   # Nodule
        27.49,   # Mass
        11.12,   # Infiltration
        2.46,    # Consolidation
        47.95,   # Emphysema
        380.80,  # Fibrosis (very rare, high weight)
        151.72,  # Pleural_Thickening (rare, high weight)
        82.00    # Hernia
    ]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    print(f"Using weighted BCE loss for class imbalance")
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Training loop
    best_auc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    
    save_dir = Path('/content/drive/MyDrive/checkpoints/vision_only')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting training for {epochs} epochs...")
    print("="*70)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(pixel_values)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  ", leave=False)
            for batch in val_pbar:
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(pixel_values)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                all_preds.append(probs.cpu())
                all_labels.append(labels.cpu())
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate AUC
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        
        from sklearn.metrics import roc_auc_score
        aucs = []
        for i in range(all_labels.shape[1]):
            if len(np.unique(all_labels[:, i])) == 2:
                try:
                    auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                    aucs.append(auc)
                except:
                    pass
        
        avg_auc = np.mean(aucs) if aucs else 0.0
        
        # Update scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_auc)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"  → Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_auc'].append(avg_auc)
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, "
              f"Val AUC={avg_auc:.4f}")
        
        # Save best model
        if avg_auc > best_auc:
            best_auc = avg_auc
            save_path = save_dir / 'vision_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'auc': avg_auc,
                'loss': avg_val_loss
            }, save_path)
            print(f"  ✓ Saved best model (AUC: {avg_auc:.4f})")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            save_path = save_dir / f'vision_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'auc': avg_auc,
                'loss': avg_val_loss
            }, save_path)
    
    print("\n" + "="*70)
    print(f"Training complete! Best AUC: {best_auc:.4f}")
    print(f"Model saved to: {save_dir / 'vision_best.pth'}")
    print("="*70)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_auc'], label='Val AUC', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Validation AUC')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=150)
    print(f"\nTraining curves saved to: {save_dir / 'training_curves.png'}")

if __name__ == '__main__':
    import numpy as np
    train_vision_model()

