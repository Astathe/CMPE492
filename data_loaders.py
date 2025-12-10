import torch
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets as torchvision_datasets

# Import the processor from your image_utils file
import torchvision.transforms as T

class MedicalImageProcessor:
    @staticmethod
    def process_pipeline(image, use_enhancement=False, use_bone_suppression=False, augment=False):
        # Ignore enhancement/suppression as images are preprocessed
        if augment:
            transforms = T.Compose([
                T.RandomRotation(10),
                T.RandomResizedCrop(512, scale=(0.8, 1.0)),
                T.ColorJitter(brightness=0.2, contrast=0.2)
            ])
            image = transforms(image)
        return image

    @staticmethod
    def get_augmentation(augment=False):
        # Wrapper to maintain compatibility with BinaryImageFolder which expects albumentations-style interface
        def wrapper(image):
            # image is numpy array
            pil_img = Image.fromarray(image)
            if augment:
                transforms = T.Compose([
                    T.RandomRotation(10),
                    T.RandomResizedCrop(512, scale=(0.8, 1.0)),
                    T.ColorJitter(brightness=0.2, contrast=0.2)
                ])
                pil_img = transforms(pil_img)
            return {'image': np.array(pil_img)}
        return wrapper

class IndianaMultimodalDataset(Dataset):
    DISEASES = ['Pneumonia', 'Cardiomegaly', 'Edema', 'Effusion', 'Atelectasis',
                'Pneumothorax', 'Nodule', 'Mass', 'Infiltration', 'Consolidation',
                'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

    def __init__(self, reports_csv, proj_csv, img_dir, processor, tokenizer, 
                 use_enhancement=True, use_bone_suppression=False, augment=False):
        
        # Merge DataFrames
        self.data = pd.merge(pd.read_csv(proj_csv), pd.read_csv(reports_csv), on='uid')
        if 'projection' in self.data.columns:
            self.data = self.data[self.data['projection'] == 'Frontal'].reset_index(drop=True)
        
        self._extract_labels()
        
        self.img_dir = Path(img_dir)
        self.processor = processor
        self.tokenizer = tokenizer
        
        # LOGIC TO SKIP PROCESSING:
        # If both are False, the pipeline in image_utils simply loads the image 
        # and converts to RGB, skipping the heavy OpenCV work.
        self.proc_flags = {
            'use_enhancement': use_enhancement,
            'use_bone_suppression': use_bone_suppression,
            'augment': augment
        }
        
        status = "Disabled" if not (use_enhancement or use_bone_suppression) else "Enabled"
        print(f"  [Dataset] Heavy Image Processing: {status}")
        print(f"  [Dataset] Augmentation: {'Enabled' if augment else 'Disabled'}")

    def _extract_labels(self):
        # Define disease patterns (some need variations to match actual text in findings)
        disease_patterns = {
            'Pneumonia': 'pneumonia',
            'Cardiomegaly': 'cardiomegaly',
            'Edema': 'edema',
            'Effusion': 'effusion',
            'Atelectasis': 'atelectasis',
            'Pneumothorax': 'pneumothorax',
            'Nodule': 'nodule',
            'Mass': 'mass',
            'Infiltration': 'infiltrat',  # Matches "infiltrate", "infiltration", "infiltrative"
            'Consolidation': 'consolidation',
            'Emphysema': 'emphysema',
            'Fibrosis': 'fibrosis',
            'Pleural_Thickening': 'pleural thickening',  # Matches "pleural thickening" (with space)
            'Hernia': 'hernia'
        }
        
        for d in self.DISEASES:
            pattern = disease_patterns.get(d, d.lower())
            self.data[f'has_{d}'] = self.data['findings'].astype(str).str.contains(
                pattern, case=False, regex=True
            ).astype(int)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        img_path = self.img_dir / row['filename']
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (512, 512))
            
        # If flags are False, this function runs very fast (just load + augment)
        image = MedicalImageProcessor.process_pipeline(image, **self.proc_flags)
        
        pixel_values = self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        
        # FIXED: Use 'indication' (clinical history) to avoid data leakage
        # The indication contains patient symptoms/clinical history, NOT the radiologist's findings
        indication = str(row.get('indication', ''))
        
        # Handle empty or useless indication values
        if not indication or indication.lower() in ['none', 'none.', 'nan', '']:
            indication = "Clinical history not provided."
        
        text = f"Clinical Indication: {indication}"
        tokenized = self.tokenizer(text, max_length=256, padding='max_length', truncation=True, return_tensors='pt')
        
        labels = torch.tensor([row[f'has_{d}'] for d in self.DISEASES], dtype=torch.float32)
        
        return {
            'pixel_values': pixel_values,
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'labels': labels
        }

class BinaryImageFolder(Dataset):
    def __init__(self, root, processor, augment=False):
        self.dataset = torchvision_datasets.ImageFolder(root)
        self.processor = processor
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        if self.augment:
            aug = MedicalImageProcessor.get_augmentation(True)
            img = np.array(img)
            img = aug(image=img)['image']
            img = Image.fromarray(img)
            
        pixel_values = self.processor(images=img, return_tensors="pt")['pixel_values'].squeeze(0)
        return pixel_values, label