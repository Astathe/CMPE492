#!/usr/bin/env python3
"""
Comprehensive evaluation script for trained multimodal medical imaging models.

Usage:
    python evaluate.py --checkpoint /path/to/model.pth --mode multimodal
    python evaluate.py --checkpoint /path/to/model.pth --mode vision_only
    python evaluate.py --checkpoint /path/to/model.pth --mode text_only
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, roc_curve, confusion_matrix, classification_report
from torch.utils.data import DataLoader, Subset
from transformers import AutoImageProcessor, AutoTokenizer

# Custom modules
from data_loaders import IndianaMultimodalDataset
from models import MultimodalDiseaseModel, VisionDiseaseModel, TextDiseaseModel

# Disease names
DISEASES = [
    'Pneumonia', 'Cardiomegaly', 'Edema', 'Effusion', 'Atelectasis',
    'Pneumothorax', 'Nodule', 'Mass', 'Infiltration', 'Consolidation',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]


def load_checkpoint(checkpoint_path, mode, device):
    """Load model from checkpoint"""
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model configuration from checkpoint or use defaults
    vision_model = 'google/siglip-base-patch16-512'
    text_model = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
    
    # Create model
    if mode == 'vision_only':
        model = VisionDiseaseModel(vision_model)
    elif mode == 'text_only':
        model = TextDiseaseModel(text_model)
    else:
        model = MultimodalDiseaseModel(vision_model, text_model)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"  Trained for {checkpoint['epoch'] + 1} epochs")
    if 'metrics' in checkpoint:
        print(f"  Checkpoint metrics: {checkpoint['metrics']}")
    
    return model


def get_predictions(model, loader, mode, device):
    """Get model predictions on a dataset"""
    all_probs = []
    all_labels = []
    
    # For multimodal, track individual heads
    all_v_probs = []
    all_t_probs = []
    all_f_probs = []
    
    print(f"\nGetting predictions on {len(loader.dataset)} samples...")
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            pv = batch['pixel_values'].to(device)
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            lbls = batch['labels'].to(device)
            
            if mode == 'vision_only':
                preds = model(pv)
                probs = torch.sigmoid(preds)
                all_probs.append(probs.cpu())
                
            elif mode == 'text_only':
                preds = model(ids, mask)
                probs = torch.sigmoid(preds)
                all_probs.append(probs.cpu())
                
            else:  # multimodal
                v_preds, t_preds, f_preds = model(pv, ids, mask)
                all_v_probs.append(torch.sigmoid(v_preds).cpu())
                all_t_probs.append(torch.sigmoid(t_preds).cpu())
                all_f_probs.append(torch.sigmoid(f_preds).cpu())
            
            all_labels.append(lbls.cpu())
    
    all_labels = torch.cat(all_labels).numpy()
    
    if mode == 'multimodal':
        return {
            'labels': all_labels,
            'vision_probs': torch.cat(all_v_probs).numpy(),
            'text_probs': torch.cat(all_t_probs).numpy(),
            'fusion_probs': torch.cat(all_f_probs).numpy()
        }
    else:
        all_probs = torch.cat(all_probs).numpy()
        return {
            'labels': all_labels,
            'probs': all_probs
        }


def calculate_metrics(y_true, y_prob, threshold=0.5):
    """Calculate comprehensive metrics"""
    metrics = {}
    
    # Per-disease AUC
    aucs = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) == 2:
            try:
                auc = roc_auc_score(y_true[:, i], y_prob[:, i])
                aucs.append(auc)
            except ValueError:
                aucs.append(np.nan)
        else:
            aucs.append(np.nan)
    
    metrics['auc_per_disease'] = aucs
    metrics['auc_macro'] = np.nanmean(aucs)
    
    # Predictions at threshold
    y_pred = (y_prob > threshold).astype(int)
    
    # Per-disease F1, Precision, Recall
    f1_scores = []
    precisions = []
    recalls = []
    
    for i in range(y_true.shape[1]):
        try:
            f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
            # Calculate precision and recall
            tp = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1))
            fp = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1))
            fn = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0))
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            f1_scores.append(f1)
            precisions.append(prec)
            recalls.append(rec)
        except:
            f1_scores.append(0)
            precisions.append(0)
            recalls.append(0)
    
    metrics['f1_per_disease'] = f1_scores
    metrics['f1_macro'] = np.mean(f1_scores)
    metrics['precision_per_disease'] = precisions
    metrics['recall_per_disease'] = recalls
    
    # Overall support (number of positive samples per disease)
    metrics['support'] = np.sum(y_true, axis=0).astype(int)
    
    return metrics


def print_results_table(metrics, model_name="Model"):
    """Print results in a nice table format"""
    print(f"\n{'='*80}")
    print(f"  {model_name} - Per-Disease Performance")
    print(f"{'='*80}")
    
    df = pd.DataFrame({
        'Disease': DISEASES,
        'AUC': [f"{x:.4f}" if not np.isnan(x) else "N/A" for x in metrics['auc_per_disease']],
        'F1': [f"{x:.4f}" for x in metrics['f1_per_disease']],
        'Precision': [f"{x:.4f}" for x in metrics['precision_per_disease']],
        'Recall': [f"{x:.4f}" for x in metrics['recall_per_disease']],
        'Support': metrics['support']
    })
    
    print(df.to_string(index=False))
    
    print(f"\n{'-'*80}")
    print(f"Overall Metrics:")
    print(f"  Macro AUC:       {metrics['auc_macro']:.4f}")
    print(f"  Macro F1:        {metrics['f1_macro']:.4f}")
    print(f"{'='*80}\n")


def plot_roc_curves(y_true, y_prob, save_path, title="ROC Curves"):
    """Plot ROC curves for all diseases"""
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    axes = axes.flatten()
    
    for i, disease in enumerate(DISEASES):
        ax = axes[i]
        
        if len(np.unique(y_true[:, i])) == 2:
            try:
                fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
                auc = roc_auc_score(y_true[:, i], y_prob[:, i])
                
                ax.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.3f}')
                ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'{disease}')
                ax.legend(loc="lower right")
                ax.grid(True, alpha=0.3)
            except:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(f'{disease}')
        else:
            ax.text(0.5, 0.5, 'Single class', ha='center', va='center')
            ax.set_title(f'{disease}')
    
    # Remove extra subplots
    for i in range(len(DISEASES), len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle(title, fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ ROC curves saved to: {save_path}")
    plt.close()


def plot_precision_recall_curves(y_true, y_prob, save_path, title="Precision-Recall Curves"):
    """Plot PR curves for all diseases"""
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    axes = axes.flatten()
    
    for i, disease in enumerate(DISEASES):
        ax = axes[i]
        
        if len(np.unique(y_true[:, i])) == 2:
            try:
                precision, recall, _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
                
                ax.plot(recall, precision, linewidth=2)
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_title(f'{disease}')
                ax.grid(True, alpha=0.3)
            except:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(f'{disease}')
        else:
            ax.text(0.5, 0.5, 'Single class', ha='center', va='center')
            ax.set_title(f'{disease}')
    
    # Remove extra subplots
    for i in range(len(DISEASES), len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle(title, fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ PR curves saved to: {save_path}")
    plt.close()


def compare_multimodal_heads(results, save_dir):
    """Compare vision, text, and fusion heads"""
    print(f"\n{'='*80}")
    print(f"  MULTIMODAL HEAD COMPARISON")
    print(f"{'='*80}\n")
    
    heads = ['vision', 'text', 'fusion']
    head_metrics = {}
    
    for head in heads:
        probs_key = f'{head}_probs'
        metrics = calculate_metrics(results['labels'], results[probs_key])
        head_metrics[head] = metrics
        print_results_table(metrics, f"{head.upper()} Head")
    
    # Create comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # AUC comparison
    ax = axes[0]
    x = np.arange(len(DISEASES))
    width = 0.25
    
    ax.bar(x - width, head_metrics['vision']['auc_per_disease'], width, label='Vision', alpha=0.8)
    ax.bar(x, head_metrics['text']['auc_per_disease'], width, label='Text', alpha=0.8)
    ax.bar(x + width, head_metrics['fusion']['auc_per_disease'], width, label='Fusion', alpha=0.8)
    
    ax.set_xlabel('Disease')
    ax.set_ylabel('AUC')
    ax.set_title('AUC Comparison by Head')
    ax.set_xticks(x)
    ax.set_xticklabels(DISEASES, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # F1 comparison
    ax = axes[1]
    ax.bar(x - width, head_metrics['vision']['f1_per_disease'], width, label='Vision', alpha=0.8)
    ax.bar(x, head_metrics['text']['f1_per_disease'], width, label='Text', alpha=0.8)
    ax.bar(x + width, head_metrics['fusion']['f1_per_disease'], width, label='Fusion', alpha=0.8)
    
    ax.set_xlabel('Disease')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score Comparison by Head')
    ax.set_xticks(x)
    ax.set_xticklabels(DISEASES, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Overall comparison
    ax = axes[2]
    overall_metrics = ['AUC', 'F1']
    vision_vals = [head_metrics['vision']['auc_macro'], head_metrics['vision']['f1_macro']]
    text_vals = [head_metrics['text']['auc_macro'], head_metrics['text']['f1_macro']]
    fusion_vals = [head_metrics['fusion']['auc_macro'], head_metrics['fusion']['f1_macro']]
    
    x_pos = np.arange(len(overall_metrics))
    width = 0.25
    
    ax.bar(x_pos - width, vision_vals, width, label='Vision', alpha=0.8)
    ax.bar(x_pos, text_vals, width, label='Text', alpha=0.8)
    ax.bar(x_pos + width, fusion_vals, width, label='Fusion', alpha=0.8)
    
    ax.set_ylabel('Score')
    ax.set_title('Overall Performance Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(overall_metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = save_dir / 'multimodal_head_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Head comparison chart saved to: {save_path}")
    plt.close()
    
    return head_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained medical imaging models')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, required=True, choices=['multimodal', 'vision_only', 'text_only'])
    parser.add_argument('--reports', type=str, default='/content/drive/MyDrive/archive/indiana_reports.csv')
    parser.add_argument('--projections', type=str, default='/content/drive/MyDrive/archive/indiana_projections.csv')
    parser.add_argument('--img_dir', type=str, default='/content/drive/MyDrive/archive/images/images_normalized')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'], help='Which split to evaluate')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save results')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_checkpoint(args.checkpoint, args.mode, device)
    
    # Prepare dataset
    print(f"\nLoading dataset...")
    processor = AutoImageProcessor.from_pretrained('google/siglip-base-patch16-512', use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')
    
    dataset = IndianaMultimodalDataset(
        args.reports, args.projections, args.img_dir,
        processor, tokenizer,
        use_enhancement=False, use_bone_suppression=False, augment=False
    )
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    import random
    random.seed(42)
    indices = list(range(total_size))
    random.shuffle(indices)
    
    if args.split == 'train':
        eval_indices = indices[:train_size]
    elif args.split == 'val':
        eval_indices = indices[train_size:train_size + val_size]
    else:  # test
        eval_indices = indices[train_size + val_size:]
    
    eval_set = Subset(dataset, eval_indices)
    print(f"Evaluating on {args.split} set: {len(eval_set)} samples")
    
    loader = DataLoader(eval_set, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    
    # Get predictions
    results = get_predictions(model, loader, args.mode, device)
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        checkpoint_name = Path(args.checkpoint).stem
        output_dir = Path(args.checkpoint).parent / f'eval_{checkpoint_name}_{args.split}'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving results to: {output_dir}")
    
    # Evaluate
    if args.mode == 'multimodal':
        # Compare all three heads
        head_metrics = compare_multimodal_heads(results, output_dir)
        
        # Plot ROC curves for each head
        for head in ['vision', 'text', 'fusion']:
            probs = results[f'{head}_probs']
            plot_roc_curves(results['labels'], probs, 
                          output_dir / f'roc_curves_{head}.png',
                          f'ROC Curves - {head.upper()} Head')
            plot_precision_recall_curves(results['labels'], probs,
                                        output_dir / f'pr_curves_{head}.png',
                                        f'Precision-Recall Curves - {head.upper()} Head')
        
        # Save detailed results
        results_df = pd.DataFrame({
            'Disease': DISEASES,
            'Vision_AUC': head_metrics['vision']['auc_per_disease'],
            'Text_AUC': head_metrics['text']['auc_per_disease'],
            'Fusion_AUC': head_metrics['fusion']['auc_per_disease'],
            'Vision_F1': head_metrics['vision']['f1_per_disease'],
            'Text_F1': head_metrics['text']['f1_per_disease'],
            'Fusion_F1': head_metrics['fusion']['f1_per_disease'],
            'Support': head_metrics['fusion']['support']
        })
        
    else:
        # Single model evaluation
        metrics = calculate_metrics(results['labels'], results['probs'])
        print_results_table(metrics, args.mode.upper())
        
        plot_roc_curves(results['labels'], results['probs'],
                       output_dir / 'roc_curves.png',
                       f'ROC Curves - {args.mode.upper()}')
        plot_precision_recall_curves(results['labels'], results['probs'],
                                    output_dir / 'pr_curves.png',
                                    f'Precision-Recall Curves - {args.mode.upper()}')
        
        results_df = pd.DataFrame({
            'Disease': DISEASES,
            'AUC': metrics['auc_per_disease'],
            'F1': metrics['f1_per_disease'],
            'Precision': metrics['precision_per_disease'],
            'Recall': metrics['recall_per_disease'],
            'Support': metrics['support']
        })
    
    # Save results CSV
    csv_path = output_dir / 'detailed_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"✓ Detailed results saved to: {csv_path}")
    
    print(f"\n{'='*80}")
    print(f"  EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"All results saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

