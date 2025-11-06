"""
Training Script - 30分钟快速训练版本
"""

import torch
import torch.nn as nn
import time
import os
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml


from model import create_transformer_model
from data import get_data_loaders
from utils import WarmupScheduler, save_model, plot_training_history


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    """Train one epoch"""
    model.train()
    total_loss = 0

    progress_bar = tqdm(train_loader, desc="Training", ncols=100)

    for batch_idx, (src, tgt) in enumerate(progress_bar):
        src, tgt = src.to(device), tgt.to(device)

        # Forward
        outputs = model(src)
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt.reshape(-1))

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            ppl = math.exp(min(loss.item(), 100))
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'PPL': f'{ppl:.2f}',
                'LR': f'{scheduler.get_lr():.6f}'
            })

    return total_loss / len(train_loader)


def evaluate(model, val_loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src, tgt in tqdm(val_loader, desc="Validation", ncols=100):
            src, tgt = src.to(device), tgt.to(device)
            outputs = model(src)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt.reshape(-1))
            total_loss += loss.item()

    return total_loss / len(val_loader)


def main():
    print("\n" + "=" * 70)
    print("🚀 Transformer Training - 30 Minutes Edition")
    print("=" * 70)

    # Load config
    with open('configs/base.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['device'])
    print(f"📱 Device: {device}")

    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Load data
    print("\n📥 Loading data...")
    train_loader, val_loader, vocab, idx2word = get_data_loaders(config)

    # Create model
    print("\n🤖 Creating model...")
    model = create_transformer_model(len(vocab), config)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        betas=(0.9, 0.98),
        eps=1e-9
    )
    scheduler = WarmupScheduler(
        optimizer,
        config['model']['d_model'],
        config['training']['warmup_steps']
    )

    # Training
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    print("\n" + "=" * 70)
    print("🎯 Training Start")
    print("=" * 70)
    print(f"⏰ Estimated: ~3 min/epoch × {config['training']['epochs']} = 30 min")
    print("=" * 70 + "\n")

    for epoch in range(config['training']['epochs']):
        print(f"\n📅 Epoch {epoch + 1}/{config['training']['epochs']}")
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        train_losses.append(train_loss)
        train_ppl = math.exp(min(train_loss, 100))

        # Validate
        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_ppl = math.exp(min(val_loss, 100))

        epoch_time = time.time() - start_time

        print(f"\n⏱️  Time: {epoch_time:.1f}s ({epoch_time / 60:.1f} min)")
        print(f"📊 Train Loss: {train_loss:.4f} | PPL: {train_ppl:.2f}")
        print(f"📊 Val Loss: {val_loss:.4f} | PPL: {val_ppl:.2f}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, config['paths']['model_save_path'])
            print(f"🏆 Best model saved!")

    # Save results
    print("\n💾 Saving results...")
    result_path = config['paths']['result_save_path']

    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Transformer Language Model - Training Results\n")
        f.write("Dataset: WikiText-2\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Final Train Loss: {train_losses[-1]:.4f} (PPL: {math.exp(min(train_losses[-1], 100)):.2f})\n")
        f.write(f"Final Val Loss: {val_losses[-1]:.4f} (PPL: {math.exp(min(val_losses[-1], 100)):.2f})\n")
        f.write(f"Best Val Loss: {best_val_loss:.4f} (PPL: {math.exp(min(best_val_loss, 100)):.2f})\n\n")
        f.write("Epoch History:\n")
        for i, (tl, vl) in enumerate(zip(train_losses, val_losses), 1):
            f.write(f"Epoch {i:2d}: Train={tl:.4f} | Val={vl:.4f}\n")



    # Plot
    plot_training_history(train_losses, val_losses, config['paths']['plot_save_path'])

    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    print(f"💾 Model: {config['paths']['model_save_path']}")
    print(f"📊 Results: {result_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()