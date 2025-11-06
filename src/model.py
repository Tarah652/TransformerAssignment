"""
Transformer Language Model - 支持 Encoder-only 和 Decoder-only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import yaml


class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力机制"""

    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: (batch_size, n_heads, seq_len, d_k)
            K: (batch_size, n_heads, seq_len, d_k)
            V: (batch_size, n_heads, seq_len, d_v)
            mask: Optional attention mask
        """
        d_k = Q.size(-1)

        # Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Linear projections and split into multiple heads
        Q = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        # Apply attention
        attn_output, attn_weights = self.attention(Q, K, V, mask)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # Final linear projection
        output = self.W_O(attn_output)
        output = self.dropout(output)

        return output, attn_weights


class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # FFN(x) = max(0, xW1 + b1)W2 + b2
        output = self.fc1(x)
        output = F.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = self.dropout(output)
        return output


class PositionalEncoding(nn.Module):
    """正弦位置编码"""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-Attention + Residual + LayerNorm
        attn_output, attn_weights = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-Forward + Residual + LayerNorm
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x, attn_weights


class TransformerDecoderLayer(nn.Module):
    """Transformer Decoder Layer"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Masked Self-Attention + Residual + LayerNorm
        attn_output, attn_weights = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-Forward + Residual + LayerNorm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x, attn_weights


class TransformerLM(nn.Module):
    """
    Transformer Language Model
    支持 Encoder-only 和 Decoder-only 两种模式
    """

    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2,
                 d_ff=256, max_len=512, dropout=0.1, model_type='encoder',
                 use_positional_encoding=True):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.use_positional_encoding = use_positional_encoding

        # Token Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional Encoding (可选)
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        else:
            self.pos_encoding = None

        # Encoder or Decoder layers
        if model_type == 'encoder':
            self.layers = nn.ModuleList([
                TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ])
        else:  # decoder
            self.layers = nn.ModuleList([
                TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ])

        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz, device):
        """生成因果掩码（仅用于Decoder）"""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, 0)
        return mask.unsqueeze(0)

    def forward(self, src, mask=None):
        batch_size, seq_len = src.size()

        # Embedding + scaling
        x = self.embedding(src) * math.sqrt(self.d_model)

        # Positional encoding (如果启用)
        if self.use_positional_encoding and self.pos_encoding is not None:
            x = self.pos_encoding(x)

        # 如果是decoder且没有提供mask，生成causal mask
        if self.model_type == 'decoder' and mask is None:
            mask = self.generate_square_subsequent_mask(seq_len, src.device)

        # Pass through layers
        for layer in self.layers:
            x, _ = layer(x, mask)

        # Output projection
        output = self.fc_out(x)

        return output


def create_transformer_model(vocab_size, config=None):
    """创建 Transformer 模型"""

    # 如果没有提供config，加载默认配置
    if config is None:
        with open('configs/base.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

    model_config = config['model']
    data_config = config['data']
    model_type = config.get('model_type', 'encoder')

    # 根据模型类型选择层数
    if model_type == 'encoder':
        n_layers = model_config['num_encoder_layers']
    else:
        n_layers = model_config['num_decoder_layers']

    # 获取位置编码设置
    use_pe = model_config.get('use_positional_encoding', True)

    # 创建模型（只创建一次！）
    model = TransformerLM(
        vocab_size=vocab_size,
        d_model=model_config['d_model'],
        n_heads=model_config['nhead'],
        n_layers=n_layers,
        d_ff=model_config['dim_feedforward'],
        max_len=data_config['max_length'] * 2,
        dropout=model_config['dropout'],
        model_type=model_type,
        use_positional_encoding=use_pe
    )

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'=' * 70}")
    print(f"🤖 {model_type.capitalize()}-only Transformer Model")
    print(f"{'=' * 70}")
    print(f"📊 Model Configuration:")
    print(f"   - Vocabulary Size: {vocab_size:,}")
    print(f"   - Embedding Dimension (d_model): {model_config['d_model']}")
    print(f"   - Number of Attention Heads: {model_config['nhead']}")
    print(f"   - Number of Layers: {n_layers}")
    print(f"   - Feed-Forward Dimension: {model_config['dim_feedforward']}")
    print(f"   - Dropout: {model_config['dropout']}")
    print(f"   - Use Positional Encoding: {use_pe}")
    print(f"\n📈 Model Statistics:")
    print(f"   - Total Parameters: {total_params:,}")
    print(f"   - Trainable Parameters: {trainable_params:,}")
    print(f"   - Model Size: {total_params * 4 / (1024 ** 2):.2f} MB")
    print(f"{'=' * 70}\n")

    return model


if __name__ == "__main__":
    # Test
    vocab_size = 1000
    config = {
        'model': {
            'd_model': 64,
            'nhead': 4,
            'num_encoder_layers': 2,
            'num_decoder_layers': 2,
            'dim_feedforward': 256,
            'dropout': 0.1,
            'use_positional_encoding': True
        },
        'data': {
            'max_length': 32
        },
        'model_type': 'encoder'
    }

    model = create_transformer_model(vocab_size, config)

    # Test forward pass
    test_input = torch.randint(0, vocab_size, (4, 16))
    output = model(test_input)
    print(f"✅ Test passed: {test_input.shape} -> {output.shape}")