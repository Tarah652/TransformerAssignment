"""
Utility Functions
"""

import torch
import matplotlib.pyplot as plt
import math
import os


class WarmupScheduler:
    """Learning rate scheduler with warmup"""

    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _get_lr(self):
        step = max(1, self.current_step)
        return self.d_model ** (-0.5) * min(
            step ** (-0.5),
            step * self.warmup_steps ** (-1.5)
        )

    def get_lr(self):
        return self._get_lr()


def save_model(model, path):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': model.vocab_size,
        'd_model': model.d_model,
    }, path)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def plot_training_history(train_losses, val_losses, save_path):
    """Plot training curves"""
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Perplexity
    plt.subplot(1, 2, 2)
    train_ppls = [math.exp(min(loss, 100)) for loss in train_losses]
    val_ppls = [math.exp(min(loss, 100)) for loss in val_losses]
    plt.plot(epochs, train_ppls, 'b-', label='Training PPL', linewidth=2)
    plt.plot(epochs, val_ppls, 'r-', label='Validation PPL', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Training and Validation Perplexity')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_text(model, vocab, idx2word, start_text, max_length=20, device='cpu'):
    """Generate text"""
    model.eval()

    words = start_text.split()
    indices = [vocab.get(word, vocab["<unk>"]) for word in words]
    indices = [vocab["<sos>"]] + indices

    input_seq = torch.tensor([indices]).to(device)
    generated = indices.copy()

    with torch.no_grad():
        for _ in range(max_length):
            output = model(input_seq)
            next_token_logits = output[0, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            if next_token == vocab["<eos>"] or next_token == vocab["<pad>"]:
                break

            generated.append(next_token)
            input_seq = torch.cat([input_seq, torch.tensor([[next_token]]).to(device)], dim=1)

    generated_words = [idx2word.get(idx, "<unk>") for idx in generated
                       if idx not in [vocab["<sos>"], vocab["<pad>"]]]
    return ' '.join(generated_words)