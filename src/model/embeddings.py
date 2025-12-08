import torch
import torch.nn as nn
from typing import Dict, Optional, Any


class ParamtatvaEmbedding(nn.Module):
    """
    Simplified Paramtatva embedding layer for inference.
    Loads pre-computed graph features from buffers instead of computing them.
    """

    def __init__(
        self,
        ptk_graph: Optional[Any],  # Kept for API compatibility, must be None
        embedding_dim: int,
        vocab_size: int,
        use_graph_features: bool = True,
        token_to_id: Optional[Dict[str, int]] = None,
        id_to_token: Optional[Dict[int, str]] = None,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.use_graph_features = use_graph_features
        self.vocab_size = vocab_size

        # Learnable embeddings for each phoneme/token
        self.phoneme_embeddings = nn.Embedding(self.vocab_size, embedding_dim)

        # Graph structure features (sutra index, position in sutra)
        if use_graph_features:
            # 14 sutras + 1 for special/null tokens
            self.sutra_embeddings = nn.Embedding(15, embedding_dim)
            # max position (usually < 10) + 1 for special/null tokens
            self.position_embeddings = nn.Embedding(11, embedding_dim)

        # Projection layer to combine features
        self.projection = nn.Linear(
            embedding_dim * (3 if use_graph_features else 1), embedding_dim
        )

        if use_graph_features:
            # Register buffers for lookups (expected to be loaded from state_dict)
            self.register_buffer(
                "_sutra_lookup", torch.zeros(self.vocab_size, dtype=torch.long)
            )
            self.register_buffer(
                "_position_lookup", torch.zeros(self.vocab_size, dtype=torch.long)
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.phoneme_embeddings.weight, mean=0.0, std=0.02)
        if self.use_graph_features:
            nn.init.normal_(self.sutra_embeddings.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)

    def forward(self, phoneme_indices: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = phoneme_indices.shape

        # Get base phoneme embeddings
        phoneme_embeds = self.phoneme_embeddings(phoneme_indices)

        if not self.use_graph_features:
            return phoneme_embeds

        # Ensure lookup tables are on the same device as input
        device = phoneme_indices.device
        if self._sutra_lookup.device != device:
            self._sutra_lookup = self._sutra_lookup.to(device)
            self._position_lookup = self._position_lookup.to(device)

        # Vectorized lookup
        sutra_indices = self._sutra_lookup[phoneme_indices]
        position_indices = self._position_lookup[phoneme_indices]

        sutra_embeds = self.sutra_embeddings(sutra_indices)
        position_embeds = self.position_embeddings(position_indices)

        # Combine all features
        combined = torch.cat([phoneme_embeds, sutra_embeds, position_embeds], dim=-1)
        output = self.projection(combined)

        return output


class PratyaharaAttentionBias(nn.Module):
    """
    Simplified Pratyahara attention bias for inference.
    """

    def __init__(
        self, ptk_graph: Optional[Any], num_heads: int, vocab_size: int = None
    ):
        super().__init__()

        self.num_heads = num_heads
        self.vocab_size = vocab_size

        # Buffer for pratyahara matrix (expected to be loaded from state_dict)
        if vocab_size:
            self.register_buffer(
                "pratyahara_matrix", torch.zeros(vocab_size, vocab_size)
            )
        else:
            # Fallback for initialization before loading weights if size unknown (should be passed)
            self.register_buffer("pratyahara_matrix", torch.zeros(1, 1))

        # Learnable scaling for bias
        self.bias_scale = nn.Parameter(torch.ones(num_heads))

    def forward(
        self, phoneme_indices: torch.Tensor, attention_scores: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len = phoneme_indices.shape
        device = attention_scores.device

        if self.pratyahara_matrix.device != device:
            # Ideally buffers move with model.to(), but strictly ensuring here
            pass

        # Check matrix size matches vocab (resize if dummy init was involved - rare in inference)
        vocab_size = len(self.pratyahara_matrix)
        phoneme_indices_clamped = torch.clamp(phoneme_indices, 0, vocab_size - 1)

        # Expand indices
        idx_i = phoneme_indices_clamped.unsqueeze(2).expand(
            batch_size, seq_len, seq_len
        )
        idx_j = phoneme_indices_clamped.unsqueeze(1).expand(
            batch_size, seq_len, seq_len
        )

        idx_i_flat = idx_i.reshape(-1)
        idx_j_flat = idx_j.reshape(-1)

        relationships_flat = self.pratyahara_matrix[idx_i_flat, idx_j_flat]
        relationships = relationships_flat.view(batch_size, seq_len, seq_len)

        bias_scale = self.bias_scale.view(1, self.num_heads, 1, 1)
        bias = relationships.unsqueeze(1) * bias_scale

        return attention_scores + bias


class MaBridgeNormalization(nn.Module):
    """
    Ma-bridge normalization layer.
    """

    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()

        self.norm = nn.LayerNorm(hidden_dim, eps=eps)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(x)
        gate_values = self.sigmoid(self.gate(x))
        output = normalized * gate_values
        return output
