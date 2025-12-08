"""
Paramtatva Transformer - Main model architecture.

This module implements a Transformer decoder with Paramtatva graph-based embeddings
instead of standard positional encodings.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
import math

from .model_configs import ModelConfig
from .embeddings import (
    ParamtatvaEmbedding,
    PratyaharaAttentionBias,
    MaBridgeNormalization
)


class TransformerBlock(nn.Module):
    """Single transformer decoder block with graph-aware attention."""
    
    def __init__(self, config: ModelConfig, ptk_graph: Optional[Any]):
        super().__init__()
        
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        
        # Multi-head self-attention
        self.attention_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Pratyahara attention bias
        if config.use_pratyahara_bias:
            self.pratyahara_bias = PratyaharaAttentionBias(ptk_graph, config.num_heads, vocab_size=config.vocab_size)
        else:
            self.pratyahara_bias = None
        
        # Feed-forward network
        self.ffn_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.ffn_up = nn.Linear(config.hidden_dim, config.intermediate_dim)
        self.ffn_down = nn.Linear(config.intermediate_dim, config.hidden_dim)
        
        # Activation
        if config.activation == "gelu":
            self.activation = nn.GELU()
        elif config.activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {config.activation}")
        
        # Dropout
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.ffn_dropout = nn.Dropout(config.dropout)
        
        # Cross-attention (if enabled)
        if config.add_cross_attention:
            self.cross_attention_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
            self.cross_q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.cross_k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.cross_v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.cross_out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
            self.cross_attention_dropout = nn.Dropout(config.attention_dropout)
        else:
            self.cross_attention_norm = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        phoneme_indices: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            attention_mask: (batch, seq_len)
            phoneme_indices: (batch, seq_len) for Pratyahara bias
        
        Returns:
            Output hidden states
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Add Pratyahara bias if enabled
        if self.pratyahara_bias is not None and phoneme_indices is not None:
            attention_scores = self.pratyahara_bias(phoneme_indices, attention_scores)
        
        # Apply causal mask (for autoregressive generation)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=hidden_states.device),
            diagonal=1
        ).bool()
        attention_scores = attention_scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]  # (batch, 1, 1, seq_len)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Compute attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, v)
        
        # Reshape back
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.hidden_dim)
        
        # Output projection
        attn_output = self.out_proj(attention_output)
        
        if torch.isnan(attn_output).any():
            print("NaN in self-attention output!")
            return hidden_states # Return input to avoid crash downstream, but already printed

        hidden_states = residual + attn_output
        
        # Cross-attention
        if self.cross_attention_norm is not None and encoder_hidden_states is not None:
            if torch.isnan(encoder_hidden_states).any():
                print("NaN in encoder_hidden_states!")
                return hidden_states

            residual = hidden_states
            norm_hidden_states = self.cross_attention_norm(hidden_states)
            
            # Project Q from decoder, K/V from encoder
            q = self.cross_q_proj(norm_hidden_states)
            k = self.cross_k_proj(encoder_hidden_states)
            v = self.cross_v_proj(encoder_hidden_states)
            
            # Reshape
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Encoder sequence length might differ
            enc_seq_len = encoder_hidden_states.shape[1]
            k = k.view(batch_size, enc_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, enc_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Compute attention scores
            attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # Apply encoder attention mask if provided
            if encoder_attention_mask is not None:
                # (batch, 1, 1, enc_seq_len)
                encoder_attention_mask = encoder_attention_mask[:, None, None, :]
                attention_scores = attention_scores.masked_fill(encoder_attention_mask == 0, float('-inf'))
            
            # Compute weights and output
            attention_weights = torch.softmax(attention_scores, dim=-1)
            
            # Handle cases where mask is all 0s (e.g. text-only samples in mixed batch)
            # This results in softmax(-inf) -> nan (0/0). We replace nan with 0.
            attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
            
            attention_weights = self.cross_attention_dropout(attention_weights)
            
            attention_output = torch.matmul(attention_weights, v)
            
            # Reshape back
            attention_output = attention_output.transpose(1, 2).contiguous()
            attention_output = attention_output.view(batch_size, seq_len, self.hidden_dim)
            
            # Output projection
            attention_output = self.cross_out_proj(attention_output)
            hidden_states = residual + attention_output
        
        # Feed-forward with residual
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn_up(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.ffn_down(hidden_states)
        hidden_states = self.ffn_dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class ParamtatvaTransformer(nn.Module):
    """
    Main Paramtatva Transformer model.
    
    Uses Maheshwara Sutras graph-based embeddings instead of standard positional encodings.
    """
    
    def __init__(
        self,
        config: ModelConfig,
        ptk_graph: Optional[Any],
        token_to_id: Optional[Dict[str, int]] = None,
        id_to_token: Optional[Dict[int, str]] = None
    ):
        """
        Initialize Paramtatva Transformer.
        
        Args:
            config: Model configuration
            ptk_graph: Paramtatva graph instance (Optional)
            token_to_id: Tokenizer's token-to-ID mapping
            id_to_token: Tokenizer's ID-to-token mapping
        """
        super().__init__()
        
        self.config = config
        self.ptk_graph = ptk_graph
        
        # Paramtatva embeddings (replaces standard token + positional embeddings)
        self.embeddings = ParamtatvaEmbedding(
            ptk_graph=ptk_graph,
            embedding_dim=config.hidden_dim,
            vocab_size=config.vocab_size, # Pass explicit vocab size
            use_graph_features=config.use_graph_embeddings,
            token_to_id=token_to_id,
            id_to_token=id_to_token
        )
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config, ptk_graph)
            for _ in range(config.num_layers)
        ])
        
        # Ma-bridge normalization (optional)
        if config.use_ma_bridge:
            self.ma_bridge = MaBridgeNormalization(config.hidden_dim)
        else:
            self.ma_bridge = None
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        
        # Output projection to vocabulary
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        # Tie weights with input embeddings
        self.lm_head.weight = self.embeddings.phoneme_embeddings.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            input_ids: Input phoneme indices (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            labels: Target labels for training (batch, seq_len)
            
        Returns:
            logits: Output logits (batch, seq_len, vocab_size)
            loss: Loss value if labels provided
        """
        # Get Paramtatva embeddings
        hidden_states = self.embeddings(input_ids)
        
        # Pass through transformer blocks
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                phoneme_indices=input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask
            )
        
        # Apply Ma-bridge normalization if enabled
        if self.ma_bridge is not None:
            hidden_states = self.ma_bridge(hidden_states)
        
        # Final layer norm
        hidden_states = self.final_norm(hidden_states)
        
        # Project to vocabulary
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
        
        return logits, loss
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Starting tokens (batch, seq_len)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            
        Returns:
            Tuple of:
                - Generated token sequences
                - Dictionary with token usage stats:
                    - 'tokens_consumed': Number of input tokens
                    - 'tokens_generated': Number of newly generated tokens
        """
        self.eval()
        
        # Track initial input length
        initial_length = input_ids.shape[1]
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                logits, _ = self.forward(input_ids)
                
                # Get logits for last position
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # Calculate token usage
        final_length = input_ids.shape[1]
        token_stats = {
            'tokens_consumed': initial_length,
            'tokens_generated': final_length - initial_length
        }
        
        return input_ids, token_stats
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    from ..paramtatva.ptk_graph import create_paramtatva_kernel
    from .model_configs import get_tiny_config
    
    # Create graph and model
    ptk = create_paramtatva_kernel()
    config = get_tiny_config(vocab_size=100)
    
    model = ParamtatvaTransformer(config, ptk)
    
    print(f"Model created with {model.get_num_params():,} parameters")
    
    # Test forward pass
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    logits, loss = model(input_ids, labels=labels)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test generation
    start_ids = torch.randint(0, config.vocab_size, (1, 5))
    generated = model.generate(start_ids, max_length=10)
    print(f"Generated shape: {generated.shape}")
