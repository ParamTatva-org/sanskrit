"""
Sanskrit Diffusion Models.

Wrappers for image and video generation using Diffusion models,
integrated with Sanskrit tokenization and concepts.
"""

import torch
import torch.nn as nn

try:
    from diffusers import (
        StableDiffusionPipeline,
        StableVideoDiffusionPipeline,
    )

    _DIFFUSERS_AVAILABLE = True
except ImportError:
    _DIFFUSERS_AVAILABLE = False

from ..tokenizer import SanskritTokenizer


class SanskritImageGenerator:
    """
    Wrapper for Text-to-Image generation using Stable Diffusion,
    customized for Sanskrit prompts.
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-2-1-base",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_float16: bool = True,
    ):
        if not _DIFFUSERS_AVAILABLE:
            raise ImportError(
                "diffusers library is required for SanskritImageGenerator"
            )

        self.device = device
        self.dtype = torch.float16 if use_float16 and device != "cpu" else torch.float32

        print(f"Loading Stable Diffusion model: {model_id}...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=self.dtype
        )
        self.pipe = self.pipe.to(self.device)

        # Initialize Sanskrit tokenizer for prompt processing
        self.tokenizer = SanskritTokenizer()

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
    ):
        """
        Generate image from Sanskrit prompt.
        """
        # Preprocess Sanskrit prompt (normalization, etc.)
        # In a real scenario, we might translate Sanskrit to English here using an LLM
        # or use a multilingual text encoder. For now, we assume the model
        # (or a multilingual generic one) can handle the tokens or we pass as is.
        # We use the tokenizer to at least normalize it.
        normalized_prompt = self.tokenizer.normalize_text(prompt)
        print(f"Processing Prompt: {normalized_prompt}")

        image = self.pipe(
            prompt=normalized_prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

        return image


class SanskritVideoGenerator:
    """
    Wrapper for Image-to-Video generation using Stable Video Diffusion.
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_float16: bool = True,
    ):
        if not _DIFFUSERS_AVAILABLE:
            raise ImportError(
                "diffusers library is required for SanskritVideoGenerator"
            )

        self.device = device
        self.dtype = torch.float16 if use_float16 and device != "cpu" else torch.float32

        print(f"Loading Stable Video Diffusion model: {model_id}...")
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=self.dtype, variant="fp16" if use_float16 else None
        )
        # Enable model cpu offload for memory saving if needed, but let's stick to explicit device
        # self.pipe.enable_model_cpu_offload()
        self.pipe = self.pipe.to(self.device)

    def generate(
        self,
        image,
        decode_chunk_size: int = 8,
        motion_bucket_id: int = 127,
        fps: int = 7,
        num_frames: int = 25,
    ):
        """
        Generate video from an input image.
        """
        frames = self.pipe(
            image,
            decode_chunk_size=decode_chunk_size,
            motion_bucket_id=motion_bucket_id,
            fps=fps,
            num_frames=num_frames,
        ).frames[0]

        return frames


class PTKGuidedDiffusion(nn.Module):
    """
    Placeholder for Paramtatva-guided diffusion.
    Future implementation could inject PTK graph features into the UNet noise prediction.
    """

    def __init__(self):
        super().__init__()
        pass
