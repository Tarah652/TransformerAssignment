"""
Data Loading Module - WikiText-2 本地读取版
数据来源：Salesforce Research
论文：Pointer Sentinel Mixture Models (Merity et al., 2017)
"""

import torch
import os
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tqdm import tqdm
import yaml


class WikiText2Dataset(Dataset):
    """
    WikiText-2 Dataset - 从本地文件读取

    Dataset Information:
    - Name: WikiText-2
    - Source: Salesforce Research
    - Paper: Pointer Sentinel Mixture Models (Merity et al., 2017)
    - Size: ~2M tokens (train), ~200K tokens (validation)
    - URL: https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/
    """

    def __init__(self, split='train', max_length=32, vocab_size=1500, vocab=None):
        self.split = split
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.texts = []

        # 如果提供了 vocab，直接使用
        if vocab is not None:
            self.vocab = vocab
            self.idx2word = {idx: word for word, idx in vocab.items()}

        print(f"\n{'=' * 70}")
        print(f"📚 Loading WikiText-2 Dataset ({split})")
        print(f"{'=' * 70}")
        print(f"📖 Dataset: WikiText-2 (Merity et al., 2017)")
        print(f"🔗 Source: Salesforce Research")

        # 加载数据
        self.load_from_file()

        # Build vocabulary (only for training and if not provided)
        if split == 'train' and vocab is None:
            self.build_vocab()

        print(f"\n📊 Dataset Statistics:")
        print(f"   - Sequences: {len(self.texts):,}")
        if hasattr(self, 'vocab'):
            print(f"   - Vocabulary: {len(self.vocab):,}")
        print(f"   - Max Length: {self.max_length}")

    def load_from_file(self):
        """从本地文件读取 WikiText-2"""

        # 文件路径
        if self.split == 'train':
            filename = 'wiki.train.tokens'
        elif self.split == 'validation':
            filename = 'wiki.valid.tokens'
        else:
            filename = 'wiki.test.tokens'

        # 查找文件的可能位置
        possible_paths = [
            f'./data/wikitext2/{filename}',
            f'../data/wikitext2/{filename}',
            f'data/wikitext2/{filename}',
        ]

        filepath = None
        for path in possible_paths:
            if os.path.exists(path):
                filepath = path
                break

        if filepath is None:
            print(f"\n❌ Error: Could not find {filename}")
            print(f"\n📥 Please download WikiText-2 manually:")
            print(f"   1. Visit: https://github.com/pytorch/examples/tree/main/word_language_model/data/wikitext-2")
            print(f"   2. Download: {filename}")
            print(f"   3. Place it in: ./data/wikitext2/{filename}")
            print(f"\n🔍 Searched in:")
            for path in possible_paths:
                print(f"   - {os.path.abspath(path)}")
            raise FileNotFoundError(f"WikiText-2 file not found: {filename}")

        print(f"📂 Reading from: {filepath}")

        # 读取文件
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 处理文本
        print(f"📖 Processing {len(lines):,} lines...")
        for line in tqdm(lines, desc="Processing", ncols=80):
            text = line.strip()
            # 过滤空行和标题行
            if len(text) > 10 and not text.startswith('=') and not text.startswith('@'):
                self.texts.append(text)

        print(f"✅ Loaded {len(self.texts):,} valid sequences")

    def build_vocab(self):
        """Build vocabulary"""
        print("\n📚 Building vocabulary...")

        word_counts = Counter()
        for text in tqdm(self.texts, desc="Counting words", ncols=80):
            words = text.lower().split()
            word_counts.update(words)

        # 特殊标记
        self.vocab = {
            "<pad>": 0,
            "<unk>": 1,
            "<sos>": 2,
            "<eos>": 3
        }

        # 添加最常见的词
        most_common = word_counts.most_common(self.vocab_size - 4)
        for word, _ in most_common:
            if len(self.vocab) < self.vocab_size:
                self.vocab[word] = len(self.vocab)

        self.idx2word = {idx: word for word, idx in self.vocab.items()}

        print(f"\n📈 Vocabulary Statistics:")
        print(f"   - Total unique words: {len(word_counts):,}")
        print(f"   - Vocabulary size: {len(self.vocab):,}")
        print(f"\n🔤 Top 10 most frequent words:")
        for i, (word, count) in enumerate(word_counts.most_common(10), 1):
            print(f"   {i:2d}. '{word}': {count:,}")

    def text_to_indices(self, text):
        """Convert text to indices"""
        words = text.lower().split()
        indices = [self.vocab.get(word, self.vocab["<unk>"]) for word in words]
        indices = [self.vocab["<sos>"]] + indices + [self.vocab["<eos>"]]

        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
            indices[-1] = self.vocab["<eos>"]
        else:
            indices.extend([self.vocab["<pad>"]] * (self.max_length - len(indices)))

        return indices

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = self.text_to_indices(text)

        input_ids = indices[:-1]
        target_ids = indices[1:]

        return torch.tensor(input_ids, dtype=torch.long), \
               torch.tensor(target_ids, dtype=torch.long)


def get_data_loaders(config=None):
    """Get data loaders for WikiText-2"""

    if config is None:
        with open('configs/base.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

    data_config = config['data']

    print("\n" + "=" * 70)
    print("📊 WikiText-2 Dataset Information")
    print("=" * 70)
    print("Name: WikiText-2")
    print("Source: Salesforce Research")
    print("Paper: Merity et al., 'Pointer Sentinel Mixture Models', ICLR 2017")
    print("URL: https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/")
    print("=" * 70)

    # Create train dataset first
    train_dataset = WikiText2Dataset(
        split='train',
        max_length=data_config['max_length'],
        vocab_size=data_config['vocab_size'],
        vocab=None  # Train builds vocab
    )

    # Create validation dataset with shared vocab
    val_dataset = WikiText2Dataset(
        split='validation',
        max_length=data_config['max_length'],
        vocab_size=data_config['vocab_size'],
        vocab=train_dataset.vocab  # Share vocab from train
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    print(f"\n{'=' * 70}")
    print("📦 DataLoader Ready")
    print(f"{'=' * 70}")
    print(f"✅ Train samples: {len(train_dataset):,}")
    print(f"✅ Val samples: {len(val_dataset):,}")
    print(f"✅ Batch size: {data_config['batch_size']}")
    print(f"✅ Vocabulary size: {len(train_dataset.vocab):,}")
    print(f"{'=' * 70}\n")

    return train_loader, val_loader, train_dataset.vocab, train_dataset.idx2word


if __name__ == "__main__":
    train_loader, val_loader, vocab, idx2word = get_data_loaders()
    print(f"✅ Test passed!")