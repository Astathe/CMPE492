#!/usr/bin/env python3
"""
Evaluate gated fusion of separately trained vision and text models
Combines predictions from pre-trained vision-only and text-only models
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from transformers import AutoImageProcessor, AutoTokenizer
from sklearn.metrics import roc_auc_score, f1_score
import random
import matplotlib.pyplot as plt

from data_loaders import IndianaMultimodalDataset
from models import VisionDiseaseModel, TextDiseaseModel

class LearnedGatedFusion(nn.Module):
    """Simple learned gating mechanism for fusion"""
    def __init__(self, num_diseases=14):
        super().__init__()
        # Per-disease gating weights
        self.gate = nn.Parameter(torch.ones(num_diseases) * 0.5)
    
    def forward(self, vision_logits, text_logits):
        # Sigmoid to get 0-1 weights
        alpha = torch.sigmoid(self.gate)
        # Fuse: alpha * vision + (1-alpha) * text
        return alpha * vision_logits + (1 - alpha) * text_logits

def calculate_metrics(y_true, y_prob):
    """Calculate AUC and F1"""
    aucs = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) == 2:
            try:
                auc = roc_auc_score(y_true[:, i], y_prob[:, i])
                aucs.append(auc)
            except:
                aucs.append(np.nan)
        else:
            aucs.append(np.nan)
    
    auc_macro = np.nanmean(aucs)
    
    y_pred = (y_prob > 0.5).astype(int)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    return auc_macro, f1_macro, aucs

def evaluate_fusion():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Paths
    vision_checkpoint = '/content/drive/MyDrive/checkpoints/vision_only/vision_best.pth'
    text_checkpoint = '/content/drive/MyDrive/checkpoints/text_only/text_best.pth'
    
    reports = '/content/drive/MyDrive/archive/indiana_reports.csv'
    projections = '/content/drive/MyDrive/archive/indiana_projections.csv'
    img_dir = '/content/drive/MyDrive/archive/images/images_normalized'
    
    print("="*70)
    print("  GATED FUSION EVALUATION")
    print("="*70)
    
    # Load models
    print("\nLoading vision model...")
    vision_model = VisionDiseaseModel('google/siglip-base-patch16-512').to(device)
    vision_ckpt = torch.load(vision_checkpoint, map_location=device, weights_only=False)
    vision_model.load_state_dict(vision_ckpt['model_state_dict'])
    vision_model.eval()
    print(f"✓ Vision model loaded (AUC: {vision_ckpt.get('auc', 'N/A')})")
    
    print("\nLoading text model...")
    text_model = TextDiseaseModel('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext').to(device)
    text_ckpt = torch.load(text_checkpoint, map_location=device, weights_only=False)
    text_model.load_state_dict(text_ckpt['model_state_dict'])
    text_model.eval()
    print(f"✓ Text model loaded (AUC: {text_ckpt.get('auc', 'N/A')})")
    
    # Load dataset
    print("\nLoading dataset...")
    processor = AutoImageProcessor.from_pretrained('google/siglip-base-patch16-512', use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')
    
    dataset = IndianaMultimodalDataset(
        reports, projections, img_dir,
        processor, tokenizer,
        use_enhancement=False,
        use_bone_suppression=False,
        augment=False
    )
    
    # Get validation set (same split as training)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    
    indices = list(range(total_size))
    random.seed(42)
    random.shuffle(indices)
    
    val_indices = indices[train_size:train_size + val_size]
    val_set = Subset(dataset, val_indices)
    val_loader = DataLoader(val_set, batch_size=32, num_workers=2, pin_memory=True)
    
    print(f"Validation samples: {len(val_set)}")
    
    # Get predictions from both models
    print("\nGetting predictions...")
    vision_preds = []
    text_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Get vision predictions
            v_logits = vision_model(pixel_values)
            v_probs = torch.sigmoid(v_logits)
            
            # Get text predictions
            t_logits = text_model(input_ids, attention_mask)
            t_probs = torch.sigmoid(t_logits)
            
            vision_preds.append(v_probs.cpu())
            text_preds.append(t_probs.cpu())
            all_labels.append(labels.cpu())
    
    vision_preds = torch.cat(vision_preds).numpy()
    text_preds = torch.cat(text_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    print("✓ Predictions collected")
    
    # Evaluate different fusion strategies
    print("\n" + "="*70)
    print("  FUSION STRATEGIES")
    print("="*70)
    
    results = {}
    
    # 1. Vision only
    v_auc, v_f1, v_aucs = calculate_metrics(all_labels, vision_preds)
    results['Vision Only'] = {'auc': v_auc, 'f1': v_f1, 'per_disease_auc': v_aucs}
    print(f"\n1. Vision Only:     AUC={v_auc:.4f}, F1={v_f1:.4f}")
    
    # 2. Text only
    t_auc, t_f1, t_aucs = calculate_metrics(all_labels, text_preds)
    results['Text Only'] = {'auc': t_auc, 'f1': t_f1, 'per_disease_auc': t_aucs}
    print(f"2. Text Only:       AUC={t_auc:.4f}, F1={t_f1:.4f}")
    
    # 3. Simple average
    avg_preds = (vision_preds + text_preds) / 2
    avg_auc, avg_f1, avg_aucs = calculate_metrics(all_labels, avg_preds)
    results['Average'] = {'auc': avg_auc, 'f1': avg_f1, 'per_disease_auc': avg_aucs}
    print(f"3. Simple Average:  AUC={avg_auc:.4f}, F1={avg_f1:.4f}")
    
    # 4. Max (optimistic)
    max_preds = np.maximum(vision_preds, text_preds)
    max_auc, max_f1, max_aucs = calculate_metrics(all_labels, max_preds)
    results['Max'] = {'auc': max_auc, 'f1': max_f1, 'per_disease_auc': max_aucs}
    print(f"4. Max (Optimistic): AUC={max_auc:.4f}, F1={max_f1:.4f}")
    
    # 5. Learned weights (optimize on validation set)
    print("\n5. Training learned gating...")
    fusion_model = LearnedGatedFusion().to(device)
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    # Convert to tensors
    v_logits_tensor = torch.from_numpy(np.log(vision_preds / (1 - vision_preds + 1e-7))).float().to(device)
    t_logits_tensor = torch.from_numpy(np.log(text_preds / (1 - text_preds + 1e-7))).float().to(device)
    labels_tensor = torch.from_numpy(all_labels).float().to(device)
    
    # Train fusion weights
    for epoch in range(100):
        optimizer.zero_grad()
        fused_logits = fusion_model(v_logits_tensor, t_logits_tensor)
        loss = criterion(fused_logits, labels_tensor)
        loss.backward()
        optimizer.step()
    
    # Get learned fusion predictions
    with torch.no_grad():
        learned_logits = fusion_model(v_logits_tensor, t_logits_tensor)
        learned_preds = torch.sigmoid(learned_logits).cpu().numpy()
    
    learned_auc, learned_f1, learned_aucs = calculate_metrics(all_labels, learned_preds)
    results['Learned Gating'] = {'auc': learned_auc, 'f1': learned_f1, 'per_disease_auc': learned_aucs}
    
    # Show learned weights
    weights = torch.sigmoid(fusion_model.gate).cpu().numpy()
    print(f"   Learned Gating:  AUC={learned_auc:.4f}, F1={learned_f1:.4f}")
    print(f"\n   Learned weights (0=all vision, 1=all vision, 0.5=equal):")
    
    diseases = ['Pneumonia', 'Cardiomegaly', 'Edema', 'Effusion', 'Atelectasis',
                'Pneumothorax', 'Nodule', 'Mass', 'Infiltration', 'Consolidation',
                'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
    
    for disease, weight in zip(diseases, weights):
        vision_weight = weight
        text_weight = 1 - weight
        print(f"     {disease:20s}: Vision={vision_weight:.3f}, Text={text_weight:.3f}")
    
    # Summary
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    
    best_method = max(results.items(), key=lambda x: x[1]['auc'])
    print(f"\nBest Fusion Method: {best_method[0]}")
    print(f"  AUC: {best_method[1]['auc']:.4f}")
    print(f"  F1:  {best_method[1]['f1']:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    # AUC comparison
    plt.subplot(1, 2, 1)
    methods = list(results.keys())
    aucs = [results[m]['auc'] for m in methods]
    bars = plt.bar(methods, aucs)
    bars[methods.index(best_method[0])].set_color('green')
    plt.ylabel('AUC')
    plt.title('Fusion Strategy Comparison - AUC')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim([0.4, 0.8])
    
    # F1 comparison
    plt.subplot(1, 2, 2)
    f1s = [results[m]['f1'] for m in methods]
    bars = plt.bar(methods, f1s)
    bars[methods.index(best_method[0])].set_color('green')
    plt.ylabel('F1 Score')
    plt.title('Fusion Strategy Comparison - F1')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    save_dir = Path('/content/drive/MyDrive/checkpoints/fusion')
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / 'fusion_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to: {save_dir / 'fusion_comparison.png'}")
    
    # Save results
    import pandas as pd
    df = pd.DataFrame({
        'Method': methods,
        'AUC': aucs,
        'F1': f1s
    })
    df.to_csv(save_dir / 'fusion_results.csv', index=False)
    print(f"Results saved to: {save_dir / 'fusion_results.csv'}")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    evaluate_fusion()

