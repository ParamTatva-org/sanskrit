"""
Multi-Modal Paramtatva Model.

Combines Vision/Video Encoders with the Paramtatva Transformer Decoder.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Union, Any
import torchvision.models as models
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights


class VideoEncoder(nn.Module):
    """
    Video Encoder using pre-trained 3D CNN (R(2+1)D).
    """
    
    def __init__(self, backbone_name: str = "r2plus1d_18", pretrained: bool = True):
        """
        Initialize Video Encoder.
        
        Args:
            backbone_name: Name of backbone ('r2plus1d_18')
            pretrained: Whether to use pre-trained weights
        """
        super().__init__()
        self.backbone_name = backbone_name
        
        if backbone_name == "r2plus1d_18":
            weights = R2Plus1D_18_Weights.DEFAULT if pretrained else None
            self.backbone = r2plus1d_18(weights=weights)
            # Remove classification head (fc)
            # R(2+1)D structure: stem -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool -> fc
            # We want features from layer4: (batch, 512, T/8, H/32, W/32)
            self.backbone = nn.Sequential(
                self.backbone.stem,
                self.backbone.layer1,
                self.backbone.layer2,
                self.backbone.layer3,
                self.backbone.layer4
            )
            self.feature_dim = 512
        else:
            raise ValueError(f"Unsupported video backbone: {backbone_name}")
            
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode video to feature sequence.
        
        Args:
            video: (batch, 3, T, H, W)
            
        Returns:
            features: (batch, seq_len, feature_dim)
        """
        # (batch, 512, T', H', W')
        features = self.backbone(video)
        
        batch_size, channels, t, h, w = features.shape
        
        # Flatten spatiotemporal dimensions: (batch, channels, t*h*w) -> (batch, t*h*w, channels)
        features = features.view(batch_size, channels, -1).permute(0, 2, 1)
        
        return features

from .model_configs import ModelConfig
from .paramtatva_transformer import ParamtatvaTransformer


class VisionEncoder(nn.Module):
    """
    Vision Encoder using pre-trained backbones (ResNet/ViT).
    """
    
    def __init__(self, backbone_name: str = "resnet50", pretrained: bool = True):
        """
        Initialize Vision Encoder.
        
        Args:
            backbone_name: Name of backbone ('resnet50', 'vit_b_16')
            pretrained: Whether to use pre-trained weights
        """
        super().__init__()
        self.backbone_name = backbone_name
        
        if backbone_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            resnet = models.resnet50(weights=weights)
            # Remove classification head (fc) and pooling
            # We want spatial features: (batch, 2048, 7, 7) -> flatten -> (batch, 49, 2048)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            self.feature_dim = 2048
            self.is_vit = False
            
        elif backbone_name == "vit_b_16":
            weights = ViT_B_16_Weights.DEFAULT if pretrained else None
            self.backbone = models.vit_b_16(weights=weights)
            self.feature_dim = 768
            self.is_vit = True
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
            
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to feature sequence.
        
        Args:
            images: (batch, 3, H, W)
            
        Returns:
            features: (batch, seq_len, feature_dim)
        """
        if self.is_vit:
            # ViT forward returns class token + patch tokens
            # We use the backbone's forward features method if available, 
            # but torchvision ViT forward returns logits.
            # We need to access the encoder directly.
            x = self.backbone._process_input(images)
            n = x.shape[0]
            
            # Expand class token
            batch_class_token = self.backbone.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            
            x = self.backbone.encoder(x)
            return x
            
        else:
            # ResNet: (batch, 2048, 7, 7)
            features = self.backbone(images)
            batch_size, channels, h, w = features.shape
            # Flatten spatial dimensions: (batch, channels, h*w) -> (batch, h*w, channels)
            features = features.view(batch_size, channels, -1).permute(0, 2, 1)
            return features


class MultiModalParamtatva(nn.Module):
    """
    Multi-Modal model combining Vision Encoder and Paramtatva Decoder.
    """
    
    def __init__(
        self,
        config: ModelConfig,
        ptk_graph: Optional[Any],
        vision_backbone: Optional[str] = None,
        video_backbone: Optional[str] = None,
        token_to_id: Optional[Dict[str, int]] = None,
        id_to_token: Optional[Dict[int, str]] = None
    ):
        """
        Initialize Multi-Modal model.
        
        Args:
            config: Model configuration
            ptk_graph: Paramtatva graph (Optional)
            vision_backbone: Name of vision backbone (for images)
            video_backbone: Name of video backbone (for videos)
            token_to_id: Tokenizer mapping
            id_to_token: Tokenizer mapping
        """
        super().__init__()
        
        if not config.add_cross_attention:
            raise ValueError("ModelConfig must have add_cross_attention=True for MultiModal model")
            
        self.config = config
        
        # Initialize encoders based on provided backbones
        self.vision_encoder = None
        self.video_encoder = None
        self.vision_projection = None
        self.video_projection = None
        
        if vision_backbone:
            self.vision_encoder = VisionEncoder(backbone_name=vision_backbone)
            self.vision_projection = nn.Linear(self.vision_encoder.feature_dim, config.hidden_dim)
            
        if video_backbone:
            self.video_encoder = VideoEncoder(backbone_name=video_backbone)
            self.video_projection = nn.Linear(self.video_encoder.feature_dim, config.hidden_dim)
            
        if not vision_backbone and not video_backbone:
            raise ValueError("Must provide either vision_backbone or video_backbone")
        
        # Paramtatva Decoder
        self.decoder = ParamtatvaTransformer(
            config,
            ptk_graph,
            token_to_id=token_to_id,
            id_to_token=id_to_token
        )
        
    def encode_visual(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        video_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode visual inputs and return projected embeddings.
        
        Args:
            pixel_values: Images (batch, 3, H, W)
            video_values: Videos (batch, 3, T, H, W)
            
        Returns:
            Visual embeddings (batch, seq_len, hidden_dim)
        """
        if video_values is not None and self.video_encoder is not None:
            visual_features = self.video_encoder(video_values)
            return self.video_projection(visual_features)
        elif pixel_values is not None and self.vision_encoder is not None:
            visual_features = self.vision_encoder(pixel_values)
            return self.vision_projection(visual_features)
        else:
            raise ValueError("Must provide either pixel_values or video_values")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        video_values: Optional[torch.Tensor] = None,
        modal_indices: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            input_ids: Text tokens (batch, seq_len)
            pixel_values: Images (visual_batch, 3, H, W) - may be smaller than batch
            video_values: Videos (visual_batch, 3, T, H, W) - may be smaller than batch
            modal_indices: Indices of samples that have visual inputs (visual_batch,)
            attention_mask: Text attention mask
            labels: Target labels
            
        Returns:
            logits, loss
        """
        batch_size = input_ids.shape[0]
        encoder_hidden_states = None
        encoder_attention_mask = None
        
        # 1. Encode visual inputs (if provided)
        if video_values is not None and self.video_encoder is not None:
            visual_features = self.video_encoder(video_values)
            visual_embeddings = self.video_projection(visual_features)
            visual_batch_size, visual_seq_len, hidden_dim = visual_embeddings.shape
            
            # Handle mixed batches
            if modal_indices is not None and visual_batch_size < batch_size:
                # Create full batch-size encoder states with zeros
                device = visual_embeddings.device
                encoder_hidden_states = torch.zeros(
                    batch_size, visual_seq_len, hidden_dim,
                    dtype=visual_embeddings.dtype,
                    device=device
                )
                # Fill in visual features at specified indices
                encoder_hidden_states[modal_indices] = visual_embeddings
                
                # Create encoder attention mask (1 for visual, 0 for text-only)
                encoder_attention_mask = torch.zeros(
                    batch_size, visual_seq_len,
                    dtype=torch.long,
                    device=device
                )
                encoder_attention_mask[modal_indices] = 1
            else:
                # All samples have video
                encoder_hidden_states = visual_embeddings
            
        elif pixel_values is not None and self.vision_encoder is not None:
            visual_features = self.vision_encoder(pixel_values)
            visual_embeddings = self.vision_projection(visual_features)
            visual_batch_size, visual_seq_len, hidden_dim = visual_embeddings.shape
            
            # Handle mixed batches
            if modal_indices is not None and visual_batch_size < batch_size:
                # Create full batch-size encoder states with zeros
                device = visual_embeddings.device
                encoder_hidden_states = torch.zeros(
                    batch_size, visual_seq_len, hidden_dim,
                    dtype=visual_embeddings.dtype,
                    device=device
                )
                # Fill in visual features at specified indices
                encoder_hidden_states[modal_indices] = visual_embeddings
                
                # Create encoder attention mask (1 for visual, 0 for text-only)
                encoder_attention_mask = torch.zeros(
                    batch_size, visual_seq_len,
                    dtype=torch.long,
                    device=device
                )
                encoder_attention_mask[modal_indices] = 1
            else:
                # All samples have images
                encoder_hidden_states = visual_embeddings
        
        # If no visual inputs provided, encoder_hidden_states remains None
        # Decoder will process as text-only (cross-attention will be skipped)
        
        # 2. Decode with optional cross-attention
        logits, loss = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask
        )
        
        return logits, loss
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def generate_caption(
        self,
        tokenizer,
        pixel_values: Optional[torch.Tensor] = None,
        video_values: Optional[torch.Tensor] = None,
        max_length: int = 50,
        temperature: float = 1.0,
    ) -> str:
        """
        Generate caption for an image or video.
        
        Args:
            tokenizer: SanskritTokenizer
            pixel_values: Image tensor (1, 3, H, W)
            video_values: Video tensor (1, 3, T, H, W)
            
        Returns:
            Generated caption string
        """
        self.eval()
        
        with torch.no_grad():
            # Encode visual input
            if video_values is not None and self.video_encoder is not None:
                visual_features = self.video_encoder(video_values)
                encoder_hidden_states = self.video_projection(visual_features)
                device = video_values.device
            elif pixel_values is not None and self.vision_encoder is not None:
                visual_features = self.vision_encoder(pixel_values)
                encoder_hidden_states = self.vision_projection(visual_features)
                device = pixel_values.device
            else:
                raise ValueError("Must provide valid visual input")
            
            # Start with BOS token
            input_ids = torch.tensor([[tokenizer.token_to_id[tokenizer.BOS_TOKEN]]], device=device)
            
            # Generate loop
            for _ in range(max_length):
                # Forward pass through decoder
                logits, _ = self.decoder(
                    input_ids=input_ids,
                    encoder_hidden_states=encoder_hidden_states
                )
                
                # Sample next token
                next_token_logits = logits[:, -1, :] / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Stop if EOS
                if next_token.item() == tokenizer.token_to_id[tokenizer.EOS_TOKEN]:
                    break
                    
                input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Decode
            return tokenizer.decode(input_ids[0].tolist())


if __name__ == "__main__":
    # Test MultiModal model
    from ..paramtatva.ptk_graph import create_paramtatva_kernel
    from .model_configs import get_tiny_config
    from ..tokenizer import SanskritTokenizer
    
    # Setup
    ptk = create_paramtatva_kernel()
    tokenizer = SanskritTokenizer()
    
    config = get_tiny_config(vocab_size=tokenizer.vocab_size)
    config.add_cross_attention = True
    
    model = MultiModalParamtatva(
        config,
        ptk,
        vision_backbone="resnet50", # Use ResNet for faster test
        token_to_id=tokenizer.token_to_id,
        id_to_token=tokenizer.id_to_token
    )
    
    print(f"MultiModal Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Dummy inputs
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    pixel_values = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    logits, _ = model(input_ids, pixel_values)
    print(f"Logits shape: {logits.shape}")
