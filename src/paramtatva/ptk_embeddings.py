"""
Paramtatva Embeddings - Graph-based embeddings for Sanskrit phonemes.

This module provides custom embedding layers that use the Paramtatva graph structure
instead of standard positional encodings.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any, List
import math


class ParamtatvaEmbedding(nn.Module):
    """
    Custom embedding layer based on Paramtatva graph positions.
    
    Instead of treating tokens as arbitrary indices, this layer embeds phonemes
    based on their position in the Maheshwara Sutras graph.
    """
    
    def __init__(
        self,
        ptk_graph: Optional[Any], # Made optional and Any to remove dependency
        embedding_dim: int,
        vocab_size: int = None, # Added explicit vocab_size

        use_graph_features: bool = True,
        token_to_id: Optional[Dict[str, int]] = None,
        id_to_token: Optional[Dict[int, str]] = None
    ):
        """
        Initialize Paramtatva embeddings.
        
        Args:
            ptk_graph: The Paramtatva graph instance (Optional)
            embedding_dim: Dimension of the embedding vectors
            vocab_size: Explicit vocabulary size (required if ptk_graph is None)
            use_graph_features: Whether to use graph structural features
            token_to_id: Tokenizer's token-to-ID mapping
            id_to_token: Tokenizer's ID-to-token mapping
        """
        super().__init__()
        
        self.ptk_graph = ptk_graph
        self.embedding_dim = embedding_dim
        self.use_graph_features = use_graph_features
        
        if ptk_graph is None:
            if vocab_size is None:
                raise ValueError("vocab_size must be provided if ptk_graph is None")
            self.vocab_size = vocab_size
            self.phoneme_to_idx = token_to_id or {}
            self.idx_to_phoneme = id_to_token or {}
        else:
            # Original logic
            if token_to_id is not None and id_to_token is not None:
                self.phoneme_to_idx = token_to_id
                self.idx_to_phoneme = id_to_token
            else:
                self.phoneme_to_idx = {}
                self.idx_to_phoneme = {}
                idx = 0
                for sutra in ptk_graph.MAHESHWARA_SUTRAS:
                    for phoneme in sutra.phonemes:
                        self.phoneme_to_idx[phoneme] = idx
                        self.idx_to_phoneme[idx] = phoneme
                        idx += 1
                    if sutra.marker:
                        self.phoneme_to_idx[sutra.marker] = idx
                        self.idx_to_phoneme[idx] = sutra.marker
                        idx += 1
            self.vocab_size = len(self.phoneme_to_idx)
        
        # Learnable embeddings for each phoneme/token
        self.phoneme_embeddings = nn.Embedding(self.vocab_size, embedding_dim)
        
        # Graph structure features (sutra index, position in sutra)
        if use_graph_features:
            # 14 sutras + 1 for special/null tokens
            self.sutra_embeddings = nn.Embedding(15, embedding_dim)
            # max position (usually < 10) + 1 for special/null tokens
            self.position_embeddings = nn.Embedding(11, embedding_dim)
        
        # Projection layer to combine features
        self.projection = nn.Linear(embedding_dim * (3 if use_graph_features else 1), embedding_dim)
        
        if use_graph_features:
            self._create_lookup_tables()
            
        self._init_weights()
    
    def _create_lookup_tables(self):
        """Create and register lookup tables for graph features."""
        # Initialize with special index (14 for sutra, 10 for position)
        sutra_lookup = torch.full((self.vocab_size,), 14, dtype=torch.long)
        position_lookup = torch.full((self.vocab_size,), 10, dtype=torch.long)
        
        if self.ptk_graph is not None:
            for idx, phoneme in self.idx_to_phoneme.items():
                pos = self.ptk_graph.get_phoneme_position(phoneme)
                if pos:
                    sutra_lookup[idx] = pos[0] - 1  # 0-indexed
                    position_lookup[idx] = pos[1]
        
        # Cache as buffers (moved to device automatically)
        self.register_buffer('_sutra_lookup', sutra_lookup)
        self.register_buffer('_position_lookup', position_lookup)

    def _init_weights(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.phoneme_embeddings.weight, mean=0.0, std=0.02)
        if self.use_graph_features:
            nn.init.normal_(self.sutra_embeddings.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
    
    def forward(self, phoneme_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get embeddings.
        
        Args:
            phoneme_indices: Tensor of shape (batch_size, seq_len) with phoneme indices
            
        Returns:
            Embeddings of shape (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len = phoneme_indices.shape
        
        # Get base phoneme embeddings
        phoneme_embeds = self.phoneme_embeddings(phoneme_indices)
        
        if not self.use_graph_features:
            return phoneme_embeds
        
        # Create lookup tables for sutra and position indices (vectorized)
        # These are initialized in __init__ via _create_lookup_tables

        
        # Ensure lookup tables are on the same device as input
        device = phoneme_indices.device
        if self._sutra_lookup.device != device:
            self._sutra_lookup = self._sutra_lookup.to(device)
            self._position_lookup = self._position_lookup.to(device)
        
        # Vectorized lookup - no loops!
        sutra_indices = self._sutra_lookup[phoneme_indices]
        position_indices = self._position_lookup[phoneme_indices]
        
        sutra_embeds = self.sutra_embeddings(sutra_indices)
        position_embeds = self.position_embeddings(position_indices)
        
        # Combine all features
        combined = torch.cat([phoneme_embeds, sutra_embeds, position_embeds], dim=-1)
        output = self.projection(combined)
        
        return output
    
    def encode_phonemes(self, phonemes: List[str]) -> torch.Tensor:
        """
        Encode a list of phonemes to indices.
        
        Args:
            phonemes: List of phoneme strings
            
        Returns:
            Tensor of phoneme indices
        """
        indices = []
        for phoneme in phonemes:
            idx = self.phoneme_to_idx.get(phoneme, 0)  # Use 0 for unknown
            indices.append(idx)
        return torch.tensor(indices, dtype=torch.long)
    
    def decode_indices(self, indices: torch.Tensor) -> List[str]:
        """
        Decode phoneme indices to strings.
        
        Args:
            indices: Tensor of phoneme indices
            
        Returns:
            List of phoneme strings
        """
        phonemes = []
        for idx in indices.tolist():
            phoneme = self.idx_to_phoneme.get(idx, '<UNK>')
            phonemes.append(phoneme)
        return phonemes


class PratyaharaAttentionBias(nn.Module):
    """
    Attention bias based on Pratyahara relationships in the graph.
    
    This module modifies attention scores to give higher weight to phonemes
    that are related through Pratyahara connections.
    """
    
    def __init__(self, ptk_graph: Optional[Any], num_heads: int, vocab_size: int = None):
        """
        Initialize Pratyahara attention bias.
        
        Args:
            ptk_graph: The Paramtatva graph instance (Optional)
            num_heads: Number of attention heads
            vocab_size: Explicit vocab size (used if ptk_graph is None)
        """
        super().__init__()
        
        self.ptk_graph = ptk_graph
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        
        # Compute pairwise Pratyahara relationships
        self.pratyahara_matrix = self._compute_pratyahara_matrix()
        
        # Learnable scaling for bias
        self.bias_scale = nn.Parameter(torch.ones(num_heads))
    
    def _compute_pratyahara_matrix(self) -> torch.Tensor:
        """
        Compute a matrix of Pratyahara relationships.
        
        Returns:
            Matrix where entry (i, j) indicates if phonemes i and j are in the same Pratyahara
        """
        if self.ptk_graph is None:
             if self.vocab_size is None:
                  # This shouldn't happen if initialized correctly, but safe fallback
                  return torch.zeros(1, 1) 
             return torch.zeros(self.vocab_size, self.vocab_size)
             
        vocab_size = len(self.ptk_graph.phoneme_to_node)
        matrix = torch.zeros(vocab_size, vocab_size)
        
        # For simplicity, we'll compute based on sutra proximity
        # Phonemes in the same or adjacent sutras have stronger relationships
        phoneme_list = list(self.ptk_graph.phoneme_to_node.keys())
        
        for i, p1 in enumerate(phoneme_list):
            pos1 = self.ptk_graph.get_phoneme_position(p1)
            if not pos1:
                continue
                
            for j, p2 in enumerate(phoneme_list):
                pos2 = self.ptk_graph.get_phoneme_position(p2)
                if not pos2:
                    continue
                
                # Same sutra: high relationship
                if pos1[0] == pos2[0]:
                    matrix[i, j] = 1.0
                # Adjacent sutras: medium relationship
                elif abs(pos1[0] - pos2[0]) == 1:
                    matrix[i, j] = 0.5
                # Within 3 sutras: weak relationship
                elif abs(pos1[0] - pos2[0]) <= 3:
                    matrix[i, j] = 0.25
        
        return matrix
    
    def forward(
        self,
        phoneme_indices: torch.Tensor,
        attention_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Add Pratyahara bias to attention scores.
        
        Args:
            phoneme_indices: Tensor of shape (batch_size, seq_len)
            attention_scores: Attention scores of shape (batch_size, num_heads, seq_len, seq_len)
            
        Returns:
            Modified attention scores with Pratyahara bias
        """
        batch_size, seq_len = phoneme_indices.shape
        device = attention_scores.device
        
        # Move pratyahara_matrix to the same device if needed
        if not isinstance(self.pratyahara_matrix, nn.Parameter):
            if not hasattr(self, '_pratyahara_matrix_device') or self._pratyahara_matrix_device != device:
                self.pratyahara_matrix = self.pratyahara_matrix.to(device)
                self._pratyahara_matrix_device = device
        
        # Vectorized relationship lookup
        # Clamp indices to valid range
        vocab_size = len(self.pratyahara_matrix)
        phoneme_indices_clamped = torch.clamp(phoneme_indices, 0, vocab_size - 1)
        
        # Create index tensors for batch-wise 2D lookup
        # Shape: (batch_size, seq_len, seq_len)
        # For each (i, j) pair in the sequence, we want pratyahara_matrix[phoneme_i, phoneme_j]
        batch_size, seq_len = phoneme_indices.shape
        
        # Expand phoneme indices for pairwise lookup
        # idx_i: (batch_size, seq_len, 1) -> (batch_size, seq_len, seq_len)
        idx_i = phoneme_indices_clamped.unsqueeze(2).expand(batch_size, seq_len, seq_len)
        # idx_j: (batch_size, 1, seq_len) -> (batch_size, seq_len, seq_len)
        idx_j = phoneme_indices_clamped.unsqueeze(1).expand(batch_size, seq_len, seq_len)
        
        # Flatten batch and sequence dims for gathering
        idx_i_flat = idx_i.reshape(-1)
        idx_j_flat = idx_j.reshape(-1)
        
        # Use the pratyahara matrix to get relationships
        # relationships_flat: (batch_size * seq_len * seq_len,)
        relationships_flat = self.pratyahara_matrix[idx_i_flat, idx_j_flat]
        
        # Reshape back to (batch_size, seq_len, seq_len)
        relationships = relationships_flat.view(batch_size, seq_len, seq_len)
        
        # Apply per-head scaling: (batch_size, 1, seq_len, seq_len) * (1, num_heads, 1, 1)
        # Result: (batch_size, num_heads, seq_len, seq_len)
        bias_scale = self.bias_scale.view(1, self.num_heads, 1, 1)
        bias = relationships.unsqueeze(1) * bias_scale
        
        return attention_scores + bias


class MaBridgeNormalization(nn.Module):
    """
    Ma-bridge normalization layer.
    
    This layer acts as a bottleneck/normalization point inspired by the Ma phoneme (म)
    which bridges the creation and dissolution phases of the Paramtatva graph.
    """
    
    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        """
        Initialize Ma-bridge normalization.
        
        Args:
            hidden_dim: Hidden dimension size
            eps: Epsilon for numerical stability
        """
        super().__init__()
        
        self.norm = nn.LayerNorm(hidden_dim, eps=eps)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Ma-bridge normalization.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            
        Returns:
            Normalized and gated output
        """
        # Normalize
        normalized = self.norm(x)
        
        # Gating mechanism (inspired by the bridge function)
        gate_values = self.sigmoid(self.gate(x))
        
        # Apply gate
        output = normalized * gate_values
        
        return output


if __name__ == "__main__":
    # Test the embeddings
    from .ptk_graph import create_paramtatva_kernel
    
    ptk = create_paramtatva_kernel()
    embedding_layer = ParamtatvaEmbedding(ptk, embedding_dim=128)
    
    print(f"Vocabulary size: {embedding_layer.vocab_size}")
    print(f"Embedding dimension: {embedding_layer.embedding_dim}")
    
    # Test encoding
    test_phonemes = ['अ', 'इ', 'उ']
    indices = embedding_layer.encode_phonemes(test_phonemes)
    print(f"Encoded phonemes {test_phonemes}: {indices}")
    
    # Test forward pass
    batch = indices.unsqueeze(0)  # Add batch dimension
    embeddings = embedding_layer(batch)
    print(f"Embeddings shape: {embeddings.shape}")
