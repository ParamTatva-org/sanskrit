"""Model package for Sanskrit LLM."""

from .paramtatva_transformer import ParamtatvaTransformer, TransformerBlock
from .model_configs import ModelConfig, get_config
from .multimodal import MultiModalParamtatva, VisionEncoder, VideoEncoder
from .embeddings import (
    ParamtatvaEmbedding,
    PratyaharaAttentionBias,
    MaBridgeNormalization,
)

# Conditional import for diffusion models (requires diffusers library)
try:
    from .diffusion_models import (  # noqa: F401
        SanskritImageGenerator,
        SanskritVideoGenerator,
        PTKGuidedDiffusion,
    )

    _DIFFUSION_AVAILABLE = True
except ImportError:
    _DIFFUSION_AVAILABLE = False

__all__ = [
    "ParamtatvaTransformer",
    "TransformerBlock",
    "ModelConfig",
    "get_config",
    "MultiModalParamtatva",
    "VisionEncoder",
    "VideoEncoder",
    "ParamtatvaEmbedding",
    "PratyaharaAttentionBias",
    "MaBridgeNormalization",
]

if _DIFFUSION_AVAILABLE:
    __all__.extend(
        ["SanskritImageGenerator", "SanskritVideoGenerator", "PTKGuidedDiffusion"]
    )
