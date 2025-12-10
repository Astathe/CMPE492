import argparse
import torch
import random
import pandas as pd
from PIL import Image
from pathlib import Path
from transformers import AutoImageProcessor, AutoTokenizer
import os

# Import your modules
from models import MultimodalDiseaseModel, SigLIPBinaryClassifier
from image_utils import MedicalImageProcessor

# The 14 Diseases
DISEASES = [
    'Pneumonia', 'Cardiomegaly', 'Edema', 'Effusion', 'Atelectasis',
    'Pneumothorax', 'Nodule', 'Mass', 'Infiltration', 'Consolidation',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# ... [Keep get_image_path_robust, get_random_multimodal_sample, get_random_binary_sample as they were] ...

def get_image_path_robust(base_dir, filename):
    path = Path(base_dir) / filename
    if path.exists(): return path
    if not path.suffix == '.png':
        png_path = path.with_suffix(path.suffix + '.png')
        if png_path.exists(): return png_path
    return None

def get_random_multimodal_sample(reports_path, proj_path, img_dir):
    try:
        reports = pd.read_csv(reports_path)
        projections = pd.read_csv(proj_path)
    except FileNotFoundError as e:
        print(f"Error loading CSVs: {e}")
        exit()

    df = pd.merge(projections, reports, on='uid')
    if 'projection' in df.columns:
        df = df[df['projection'] == 'Frontal']
    
    for _ in range(10):
        row = df.sample(1).iloc[0]
        image_path = get_image_path_robust(img_dir, row['filename'])
        if image_path:
            # Use 'indication' to match the retrained model (avoid data leakage)
            indication = str(row.get('indication', ''))
            if indication.lower() == 'none.' or indication.lower() == 'nan':
                indication = "No clinical history provided."
            
            full_text = f"Indication: {indication}"
            
            # For "Actual", we still look at findings/impression to see the ground truth
            findings = str(row.get('findings', ''))
            impression = str(row.get('impression', ''))
            combined_truth = (findings + " " + impression).lower()
            
            actual = [d for d in DISEASES if d.lower() in combined_truth]
            return image_path, full_text, actual
    print("Error: No images found.")
    exit()

def get_random_binary_sample(dataset_dir):
    test_dir = Path(dataset_dir) / 'test'
    all_images = list(test_dir.glob('*/*'))
    if not all_images: return None, None
    choice = random.choice(all_images)
    return choice, choice.parent.name

# ==========================================
# MODEL LOADERS
# ==========================================

def load_multimodal_model(checkpoint_path, device):
    print(f"Loading Multimodal Model from: {checkpoint_path}")
    # Initialize the 3-Head Model
    model = MultimodalDiseaseModel(
        'google/siglip-base-patch16-512', 
        'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
    ).to(device)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        print(f"\nERROR: Shape mismatch!")
        print("This likely means you are trying to load an OLD checkpoint into the NEW 3-head model.")
        print("You must re-train the model using 'main.py' before running this demo.\n")
        exit()
        
    model.eval()
    return model

def load_binary_model(checkpoint_path, device):
    print(f"Loading Binary Model from: {checkpoint_path}")
    model = SigLIPBinaryClassifier('google/siglip-base-patch16-512').to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# ==========================================
# PREDICTION LOGIC (3-HEAD)
# ==========================================

def predict_multimodal(model, image_path, report_text, actual_diseases, device):
    # 1. Image
    raw_image = Image.open(image_path).convert('RGB')
    
    # Apply Pipeline (Enhancement -> Bone Suppression)
    # This is critical if using original images!
    processed_image = MedicalImageProcessor.process_pipeline(
        raw_image, 
        use_enhancement=True, 
        use_bone_suppression=False, 
        augment=False
    )
    processed_img = MedicalImageProcessor.process_pipeline(
        raw_image, use_enhancement=False, use_bone_suppression=False, augment=False
    )
    processor = AutoImageProcessor.from_pretrained('google/siglip-base-patch16-512')
    pixel_values = processor(images=processed_img, return_tensors="pt")['pixel_values'].to(device)
    
    # 2. Text
    tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext')
    tokens = tokenizer(report_text, max_length=256, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)
    
    # 3. Inference - UNPACK ALL 3 HEADS
    with torch.no_grad():
        v_logits, t_logits, f_logits = model(pixel_values, input_ids, attention_mask)
        
        # Convert all to percentages
        v_probs = torch.sigmoid(v_logits).cpu().numpy()[0] * 100
        t_probs = torch.sigmoid(t_logits).cpu().numpy()[0] * 100
        f_probs = torch.sigmoid(f_logits).cpu().numpy()[0] * 100
        
    # 4. Report
    print(f"\n{'='*95}")
    print(f" MULTIMODAL DIAGNOSIS REPORT")
    print(f"{'='*95}")
    print(f"Image Source: {image_path.name}")
    print(f"Actual Diseases: {', '.join(actual_diseases) if actual_diseases else 'Healthy'}")
    print("-" * 95)
    print(f"Input Text: \"{report_text[:120]}...\"") 
    print("-" * 95)
    
    # 3-COLUMN TABLE
    print(f"{'DISEASE':<20} | {'IMAGE %':<8} | {'TEXT %':<8} | {'FUSION %':<10} | {'STATUS'}")
    print("-" * 95)
    
    for i, disease in enumerate(DISEASES):
        v = v_probs[i]
        t = t_probs[i]
        f = f_probs[i]
        
        # Status based on Fusion
        status = ""
        if f > 50.0:
            status = "POSITIVE"
            if disease in actual_diseases: status += " [CORRECT]"
        elif disease in actual_diseases:
            status = "MISSED"

        # Show row if ANY model is confident OR if disease is actually present
        if f > 20.0 or v > 30.0 or t > 30.0 or disease in actual_diseases:
            print(f"{disease:<20} | {v:5.1f}%   | {t:5.1f}%   | {f:6.1f}%     | {status}")
            
    print("="*95 + "\n")

def predict_binary(model, image_path, actual_label, device):
    # (Binary prediction logic remains simple/unchanged)
    raw_image = Image.open(image_path).convert('RGB')
    processed_img = MedicalImageProcessor.process_pipeline(
        raw_image, use_enhancement=False, use_bone_suppression=False, augment=False
    )
    processor = AutoImageProcessor.from_pretrained('google/siglip-base-patch16-512')
    pixel_values = processor(images=processed_img, return_tensors="pt")['pixel_values'].to(device)
    
    with torch.no_grad():
        logits = model(pixel_values)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
    p_pneumonia = probs[1] * 100
    print(f"\n{'='*60}")
    print(f" PNEUMONIA CHECK (Binary)")
    print(f"{'='*60}")
    print(f"Image: {image_path.name}")
    print(f"Ground Truth: {actual_label}")
    print(f"Model Confidence: {p_pneumonia:.2f}%")
    prediction = "PNEUMONIA" if p_pneumonia > 50 else "NORMAL"
    print(f">> Prediction: {prediction} [{'CORRECT' if prediction==actual_label else 'INCORRECT'}]")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['multimodal', 'binary'], default='multimodal')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/multimodal/best_model.pth')
    parser.add_argument('--random', action='store_true', help="Pick a random image from dataset")
    parser.add_argument('--image', type=str, help="Specific image path")
    parser.add_argument('--text', type=str, default="Findings: No acute abnormalities.", help="Manual text input")
    
    # PATHS
    parser.add_argument('--reports', default='archive/indiana_reports.csv')
    parser.add_argument('--projections', default='archive/indiana_projections.csv')
    parser.add_argument('--img_dir', default='archive/images/images_normalized')
    parser.add_argument('--binary_dir', default='pneumonia_dataset')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Resolve Input
    image_path = args.image
    text_input = args.text
    actual_labels = [] 
    
    if args.random:
        if args.mode == 'multimodal':
            image_path, text_input, actual_labels = get_random_multimodal_sample(args.reports, args.projections, args.img_dir)
        else:
            image_path, actual_labels = get_random_binary_sample(args.binary_dir)
            
    if not image_path:
        print("Error: Please provide --image PATH or use --random")
        exit()
        
    if args.mode == 'multimodal':
        model = load_multimodal_model(args.checkpoint, device)
        predict_multimodal(model, image_path, text_input, actual_labels, device)
    else:
        model = load_binary_model(args.checkpoint, device)
        predict_binary(model, image_path, actual_labels, device)