"""Layer-by-layer with REAL IMAGE"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from modeling_clip import DualCLIPVisionEncoder
from transformers import CLIPVisionModel, CLIPImageProcessor
from PIL import Image
import requests
from io import BytesIO

device = torch.device('cuda')

# Load models
model_name = 'openai/clip-vit-large-patch14-336'
teacher = CLIPVisionModel.from_pretrained(model_name).to(device)
student = DualCLIPVisionEncoder.from_pretrained(model_name, attention_mode='cross').to(device)
teacher.eval()
student.eval()

# REAL IMAGE
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
pil_image = Image.open(BytesIO(requests.get(url).content))
processor = CLIPImageProcessor.from_pretrained(model_name)
inputs = processor(images=pil_image, return_tensors="pt")
image = inputs['pixel_values'].to(device)

print(f"Image shape: {image.shape}\n")

with torch.no_grad():
    # Teacher embeddings
    t_patch = teacher.vision_model.embeddings.patch_embedding(image)
    t_patch = t_patch.flatten(2).transpose(1, 2)
    t_cls = teacher.vision_model.embeddings.class_embedding.expand(1, 1, -1)
    t_emb = torch.cat([t_cls, t_patch], dim=1)
    t_input = t_emb + teacher.vision_model.embeddings.position_embedding.weight
    
    # Student embeddings  
    s_patch = student.patch_embedding(image)
    s_cls = student.cls_token.expand(1, -1, -1)
    s_emb = torch.cat([s_cls, s_patch], dim=1)
    s_input = s_emb + student.position_embeddings
    
    print(f"Input diff: {torch.abs(t_input - s_input).mean():.10f}")
    
    # Right branch
    right_input = torch.zeros_like(s_input)
    seq_len = right_input.shape[1]
    right_input = right_input + student.position_embedding_mot(torch.arange(seq_len, device=device).unsqueeze(0))
    
    # Through layers
    t_hidden = t_input
    s_left = s_input
    s_right = right_input
    
    t_hidden = teacher.vision_model.pre_layrnorm(t_hidden)
    s_left = student.pre_layrnorm(s_left)
    s_right = student.pre_layrnorm_mot(s_right)
    
    for i in range(len(teacher.vision_model.encoder.layers)):
        t_hidden = teacher.vision_model.encoder.layers[i](t_hidden, None, None)[0]
        s_left, s_right = student.layers[i](s_left, s_right)
        diff = torch.abs(t_hidden - s_left).mean().item()
        print(f"Layer {i}: diff = {diff:.10f}")
    
    # Final
    t_pooled = t_hidden[:, 0]
    t_final = teacher.vision_model.post_layernorm(t_pooled)
    t_final_dir = teacher(image).pooler_output
    
    s_pooled = s_left[:, 0]
    s_final = student.final_layernorm(s_pooled)
    
    print(f"\nFinal pooled diff: {torch.abs(t_final - s_final).mean():.10f}")
    print(f"Final pooled diff dir: {torch.abs(t_final_dir - s_final).mean():.10f}")
    print(f"Teacher final[:5]: {t_final[0,:5]}")
    print(f"Student final[:5]: {s_final[0,:5]}")
