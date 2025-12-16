"""
Adaptation Training Script for Dual Vision Encoder

This script trains the right branch (MoT) of the dual vision encoder to align
its output with the original vision encoder output, while keeping the left
branch (image) frozen.

Training objective:
- Input: Same image to both left and right branches
- Left branch: Processes image patches (frozen)
- Right branch: Processes random tokens (trainable)
- Loss: Align right branch output with teacher encoder output
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json
from typing import Optional, Literal

# Import our dual vision encoder
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DualVisionConfig
from utils import create_dual_encoder


class ImageDataset(Dataset):
    """Simple image dataset for training."""
    
    def __init__(self, image_dir: str, transform=None):
        self.image_dir = Path(image_dir)
        self.image_paths = []
        for ext in ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG']:
            self.image_paths.extend(self.image_dir.rglob(ext))
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform.preprocess(image)['pixel_values'][0]
        
        return image


def create_teacher_encoder(encoder_type: str, pretrained_model: str):
    """Create teacher encoder (original vision encoder)."""
    if encoder_type == 'vit':
        from transformers import ViTModel
        teacher = ViTModel.from_pretrained(pretrained_model)
    elif encoder_type == 'clip':
        from transformers import CLIPVisionModel
        teacher = CLIPVisionModel.from_pretrained(pretrained_model)
    elif encoder_type == 'siglip':
        from transformers import SiglipVisionModel
        teacher = SiglipVisionModel.from_pretrained(pretrained_model)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    # Freeze teacher
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()
    
    return teacher


def compute_alignment_loss(
    student_output: torch.Tensor,
    teacher_output: torch.Tensor,
    loss_type: Literal['cosine', 'mse', 'combined'] = 'combined'
) -> torch.Tensor:
    """Compute alignment loss between student and teacher outputs.
    
    Args:
        student_output: Right branch output [batch_size, hidden_size]
        teacher_output: Teacher encoder output [batch_size, hidden_size]
        loss_type: Type of loss to use
        
    Returns:
        Loss value
    """
    if loss_type == 'cosine':
        # Cosine similarity loss (maximize similarity)
        cosine_sim = F.cosine_similarity(student_output, teacher_output, dim=-1)
        loss = 1 - cosine_sim.mean()
    elif loss_type == 'mse':
        # MSE loss
        loss = F.mse_loss(student_output, teacher_output)
    elif loss_type == 'combined':
        # Combined loss
        cosine_loss = 1 - F.cosine_similarity(student_output, teacher_output, dim=-1).mean()
        mse_loss = F.mse_loss(student_output, teacher_output)
        loss = cosine_loss + 0.1 * mse_loss
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return loss


def train_one_epoch(
    model,
    teacher,
    dataloader,
    optimizer,
    device,
    loss_type: str = 'combined',
    encoder_type: str = 'vit'
):
    """Train for one epoch."""
    model.train()
    teacher.eval()
    
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        images = batch.to(device)
        
        # Forward through student (dual encoder)
        with torch.cuda.amp.autocast():
            outputs = model(images, return_dict=True)
            student_output = outputs['pooled_output']  # Right branch output by default
        
        # Forward through teacher
        with torch.no_grad():
            if encoder_type == 'vit':
                teacher_outputs = teacher(images)
                teacher_output = teacher_outputs.last_hidden_state[:, 0]  # CLS token
            elif encoder_type == 'clip':
                teacher_outputs = teacher(images)
                teacher_output = teacher_outputs.pooler_output
            elif encoder_type == 'siglip':
                teacher_outputs = teacher(images)
                teacher_output = teacher_outputs.pooler_output
        
        # Compute loss
        loss = compute_alignment_loss(student_output, teacher_output, loss_type)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': total_loss / num_batches})
    
    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, teacher, dataloader, device, loss_type: str, encoder_type: str):
    """Evaluate model."""
    model.eval()
    teacher.eval()
    
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch.to(device)
        
        # Forward through student
        outputs = model(images, return_dict=True)
        student_output = outputs['pooled_output']
        
        # Forward through teacher
        if encoder_type == 'vit':
            teacher_outputs = teacher(images)
            teacher_output = teacher_outputs.last_hidden_state[:, 0]
        elif encoder_type == 'clip':
            teacher_outputs = teacher(images)
            teacher_output = teacher_outputs.pooler_output
        elif encoder_type == 'siglip':
            teacher_outputs = teacher(images)
            teacher_output = teacher_outputs.pooler_output
        
        # Compute loss
        loss = compute_alignment_loss(student_output, teacher_output, loss_type)
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description="Train Dual Vision Encoder Adaptation")
    
    # Model settings
    parser.add_argument('--encoder_type', type=str, default='clip',
                       choices=['vit', 'clip', 'siglip'],
                       help='Type of vision encoder')
    parser.add_argument('--pretrained_model', type=str, required=True,
                       help='Pretrained model name or path')
    
    # Data settings
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to image directory')
    parser.add_argument('--val_data_path', type=str, default=None,
                       help='Path to validation image directory')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    
    # Training settings
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate for right branch')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=1,
                       help='Number of warmup epochs')
    parser.add_argument('--loss_type', type=str, default='combined',
                       choices=['cosine', 'mse', 'combined'],
                       help='Type of alignment loss')
    
    # Random token settings
    parser.add_argument('--auxiliary_token_init', type=str, default='gaussian',
                       choices=['gaussian', 'uniform'],
                       help='Distribution for random auxiliary tokens')
    parser.add_argument('--random_token_std', type=float, default=0.02,
                       help='Standard deviation for random tokens')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--save_every', type=int, default=1,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--log_every', type=int, default=100,
                       help='Log every N steps')
    
    # System settings
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create student (dual encoder) first to get preprocessor
    print(f"Creating dual vision encoder...")
    student = create_dual_encoder(
        encoder_type=args.encoder_type,
        pretrained_path=args.pretrained_model,
        auxiliary_token_init=args.auxiliary_token_init,
        random_token_std=args.random_token_std,
        output_mode='right',  # Use right branch output
    )
    
    # Get preprocessor from encoder (matches pretrained model)
    print("Using preprocessor from pretrained model...")
    transform = student.preprocessor
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = ImageDataset(args.data_path, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = None
    if args.val_data_path:
        val_dataset = ImageDataset(args.val_data_path, transform=transform)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    print(f"Training samples: {len(train_dataset)}")
    if val_loader:
        print(f"Validation samples: {len(val_dataset)}")
    
    # Create teacher encoder
    print(f"Creating teacher encoder: {args.encoder_type}")
    teacher = create_teacher_encoder(args.encoder_type, args.pretrained_model)
    teacher = teacher.to(device)
    
    student = student.to(device)
    
    # Freeze left branch
    print("Freezing left branch parameters...")
    student.freeze_left_branch()
    
    # Count parameters
    total_params = sum(p.numel() for p in student.parameters())
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters (right branch): {trainable_params:,}")
    
    # Create optimizer
    optimizer = AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.lr * 0.1
    )

    if val_loader:
        val_loss = evaluate(
            model=student,
            teacher=teacher,
            dataloader=val_loader,
            device=device,
            loss_type=args.loss_type,
            encoder_type=args.encoder_type
        )
        print(f"Val loss: {val_loss:.4f}")
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        torch.cuda.empty_cache()
        
        # Train
        train_loss = train_one_epoch(
            model=student,
            teacher=teacher,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_type=args.loss_type,
            encoder_type=args.encoder_type
        )
        
        print(f"Train loss: {train_loss:.4f}")
        
        # Validate
        if val_loader:
            val_loss = evaluate(
                model=student,
                teacher=teacher,
                dataloader=val_loader,
                device=device,
                loss_type=args.loss_type,
                encoder_type=args.encoder_type
            )
            print(f"Val loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    student.state_dict(),
                    os.path.join(args.output_dir, 'best_model.pt')
                )
                print(f"Saved best model with val loss: {val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
        
        # Step scheduler
        scheduler.step()
    
    # Save final model
    final_path = os.path.join(args.output_dir, 'final_model.pt')
    torch.save(student.state_dict(), final_path)
    print(f"\nTraining complete! Final model saved to: {final_path}")


if __name__ == '__main__':
    main()
