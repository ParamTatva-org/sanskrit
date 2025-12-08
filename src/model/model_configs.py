"""
Model configuration presets for different sizes.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for Paramtatva Transformer model."""
    
    # Model architecture
    vocab_size: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    intermediate_dim: int
    max_seq_length: int
    
    # Dropout and regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_drop: float = 0.0
    
    # Activation
    activation: str = "gelu"
    
    # Normalization
    layer_norm_eps: float = 1e-6
    
    # Paramtatva-specific
    use_graph_embeddings: bool = True
    use_pratyahara_bias: bool = True
    use_ma_bridge: bool = True
    
    # Multi-modal
    add_cross_attention: bool = False
    vision_config: Optional[dict] = None
    video_config: Optional[dict] = None
    
    # Training
    initializer_range: float = 0.02
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.hidden_dim % self.num_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"


# Predefined configurations for different model sizes

def get_tiny_config(vocab_size: int = 5000) -> ModelConfig:
    """
    Tiny model configuration (~1M parameters).
    
    For quick experiments and testing.
    """
    return ModelConfig(
        vocab_size=vocab_size,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        intermediate_dim=512,
        max_seq_length=512,
        dropout=0.1,
        attention_dropout=0.1,
    )


def get_small_config(vocab_size: int = 5000) -> ModelConfig:
    """
    Small model configuration (~10M parameters).
    
    For development and medium-scale experiments.
    """
    return ModelConfig(
        vocab_size=vocab_size,
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        intermediate_dim=1024,
        max_seq_length=1024,
        dropout=0.1,
        attention_dropout=0.1,
    )


def get_medium_config(vocab_size: int = 5000) -> ModelConfig:
    """
    Medium model configuration (~100M parameters).
    
    Similar to small GPT models.
    """
    return ModelConfig(
        vocab_size=vocab_size,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        intermediate_dim=3072,
        max_seq_length=2048,
        dropout=0.1,
        attention_dropout=0.1,
        layer_drop=0.1,
    )


def get_large_config(vocab_size: int = 5000) -> ModelConfig:
    """
    Large model configuration (~1B parameters).
    
    For serious training runs.
    """
    return ModelConfig(
        vocab_size=vocab_size,
        hidden_dim=2048,
        num_layers=24,
        num_heads=16,
        intermediate_dim=8192,
        max_seq_length=4096,
        dropout=0.1,
        attention_dropout=0.1,
        layer_drop=0.2,
    )


def get_xlarge_config(vocab_size: int = 5000) -> ModelConfig:
    """
    Extra large model configuration (~10B parameters).
    
    Requires significant compute resources.
    """
    return ModelConfig(
        vocab_size=vocab_size,
        hidden_dim=4096,
        num_layers=48,
        num_heads=32,
        intermediate_dim=16384,
        max_seq_length=8192,
        dropout=0.1,
        attention_dropout=0.1,
        layer_drop=0.3,
    )


def get_config(size: str, vocab_size: int = 5000) -> ModelConfig:
    """
    Get configuration by name.
    
    Args:
        size: Model size, one of: tiny, small, medium, large, xlarge
        vocab_size: Vocabulary size
        
    Returns:
        ModelConfig instance
    """
    size = size.lower()
    
    if size == "tiny":
        return get_tiny_config(vocab_size)
    elif size == "small":
        return get_small_config(vocab_size)
    elif size == "medium":
        return get_medium_config(vocab_size)
    elif size == "large":
        return get_large_config(vocab_size)
    elif size == "xlarge":
        return get_xlarge_config(vocab_size)
    else:
        raise ValueError(
            f"Unknown model size: {size}. "
            f"Choose from: tiny, small, medium, large, xlarge"
        )


if __name__ == "__main__":
    # Print configurations
    configs = ["tiny", "small", "medium", "large", "xlarge"]
    
    for size in configs:
        config = get_config(size)
        
        # Estimate parameters (rough)
        embedding_params = config.vocab_size * config.hidden_dim
        layer_params = (
            # Attention: Q, K, V projections + output
            4 * config.hidden_dim * config.hidden_dim +
            # FFN: 2 linear layers
            2 * config.hidden_dim * config.intermediate_dim
        ) * config.num_layers
        total_params = embedding_params + layer_params
        
        print(f"\n{size.upper()} Configuration:")
        print(f"  Layers: {config.num_layers}")
        print(f"  Hidden dim: {config.hidden_dim}")
        print(f"  Heads: {config.num_heads}")
        print(f"  Intermediate dim: {config.intermediate_dim}")
        print(f"  Estimated parameters: {total_params / 1e6:.1f}M")
