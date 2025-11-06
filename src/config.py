"""
Configuration Module
"""

import os
import yaml
import torch


def load_config():
    """Load configuration from base.yaml"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "configs", "base.yaml")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ Configuration loaded from {config_path}")
        return config
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return get_default_config()


def get_default_config():
    """Default configuration"""
    return {
        "data": {
            "path": "./data/",
            "batch_size": 32,
            "max_length": 32,
            "vocab_size": 2000
        },
        "model": {
            "d_model": 64,
            "nhead": 4,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "dim_feedforward": 256,
            "dropout": 0.1
        },
        "training": {
            "epochs": 10,
            "learning_rate": 0.0005,
            "warmup_steps": 500
        },
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "paths": {
            "model_save_path": "./models/transformer_best.pth",
            "result_save_path": "./results/training_results.txt"
        },
        "model_type": "encoder"
    }


# Global config object
config = load_config()