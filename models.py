import torch
import torch.nn as nn
from transformers import SiglipVisionModel, AutoModel

# ---------------------------------------------------------
# 1. VISION-ONLY MODEL (14 Diseases)
# ---------------------------------------------------------
class VisionDiseaseModel(nn.Module):
    def __init__(self, model_name='google/siglip-base-patch16-512', num_diseases=14, freeze_backbone=False):
        super().__init__()
        try:
            self.vision_model = SiglipVisionModel.from_pretrained(model_name)
        except:
            full_model = AutoModel.from_pretrained(model_name)
            self.vision_model = full_model.vision_model
            
        self.hidden_size = getattr(self.vision_model.config, 'hidden_size', 768)
        
        if freeze_backbone:
            for param in self.vision_model.parameters():
                param.requires_grad = False
                
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_diseases)
        )

    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        # Use pooler_output if available, else mean pool
        pooled = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state.mean(dim=1)
        return self.classifier(pooled)

# ---------------------------------------------------------
# 2. TEXT-ONLY MODEL (14 Diseases)
# ---------------------------------------------------------
class TextDiseaseModel(nn.Module):
    def __init__(self, model_name='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext', num_diseases=14):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.text_model.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_diseases)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0, :]
        return self.classifier(pooled)

# ---------------------------------------------------------
# 3. MULTIMODAL MODEL (3-Heads)
# ---------------------------------------------------------
class MultimodalDiseaseModel(nn.Module):
    def __init__(self, vision_name, text_name, num_diseases=14, fusion_method='gated'):
        super().__init__()
        
        # Encoders
        self.vision_model = SiglipVisionModel.from_pretrained(vision_name)
        self.text_model = AutoModel.from_pretrained(text_name)
        
        v_dim = self.vision_model.config.hidden_size
        t_dim = self.text_model.config.hidden_size
        
        # Heads
        self.vision_head = nn.Sequential(nn.Linear(v_dim, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_diseases))
        self.text_head = nn.Sequential(nn.Linear(t_dim, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, num_diseases))
        
        # Fusion
        self.fusion_method = fusion_method
        self.v_proj = nn.Linear(v_dim, 512)
        self.t_proj = nn.Linear(t_dim, 512)
        if fusion_method == 'gated':
            self.gate = nn.Sequential(nn.Linear(1024, 512), nn.Sigmoid())
            
        self.fusion_head = nn.Sequential(
            nn.Linear(512 if fusion_method=='gated' else (v_dim+t_dim), 256),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_diseases)
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        # Features
        v_out = self.vision_model(pixel_values=pixel_values)
        v_feat = v_out.pooler_output if hasattr(v_out, 'pooler_output') else v_out.last_hidden_state.mean(dim=1)
        
        t_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        t_feat = t_out.pooler_output if hasattr(t_out, 'pooler_output') else t_out.last_hidden_state[:, 0, :]
        
        # Individual Heads
        v_logits = self.vision_head(v_feat)
        t_logits = self.text_head(t_feat)
        
        # Fusion
        if self.fusion_method == 'gated':
            vp = self.v_proj(v_feat)
            tp = self.t_proj(t_feat)
            gate = self.gate(torch.cat([vp, tp], dim=1))
            fused = gate * vp + (1 - gate) * tp
        else:
            fused = torch.cat([v_feat, t_feat], dim=1)
            
        f_logits = self.fusion_head(fused)
        return v_logits, t_logits, f_logits

# ---------------------------------------------------------
# 4. BINARY CLASSIFIER (Pneumonia vs Normal)
# ---------------------------------------------------------
class SigLIPBinaryClassifier(nn.Module):
    def __init__(self, model_name='google/siglip-base-patch16-512'):
        super().__init__()
        try:
            self.vision_model = SiglipVisionModel.from_pretrained(model_name)
        except:
            full_model = AutoModel.from_pretrained(model_name)
            self.vision_model = full_model.vision_model
            
        self.hidden_size = getattr(self.vision_model.config, 'hidden_size', 768)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size, 2) # Binary: 0=Normal, 1=Pneumonia
        )

    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state.mean(dim=1)
        return self.classifier(pooled)