import argparse
import os
import sys
from PIL import Image

# Add current directory to sys.path
sys.path.insert(0, os.getcwd())


try:
    from src.model.diffusion_models import (
        SanskritImageGenerator,
        SanskritVideoGenerator,
    )
except ImportError:
    SanskritImageGenerator = None
    SanskritVideoGenerator = None


def main():
    parser = argparse.ArgumentParser(description="Run Sanskrit Diffusion Inference")
    subparsers = parser.add_subparsers(dest="mode", help="Mode: image or video")

    # Image Generation Args
    img_parser = subparsers.add_parser("image", help="Generate Image from Text")
    img_parser.add_argument("--prompt", type=str, required=True, help="Sanskrit prompt")
    img_parser.add_argument(
        "--output", type=str, default="output_image.png", help="Output filename"
    )
    img_parser.add_argument("--steps", type=int, default=30, help="Inference steps")

    # Video Generation Args
    vid_parser = subparsers.add_parser("video", help="Generate Video from Image")
    vid_parser.add_argument("--image", type=str, required=True, help="Input image path")
    vid_parser.add_argument(
        "--output", type=str, default="output_video.mp4", help="Output folder/filename"
    )

    args = parser.parse_args()

    if args.mode == "image":
        print("Initializing Image Generator...")
        if SanskritImageGenerator is None:
            print(
                "Error: Diffusion models not available. Install 'torch' and 'diffusers'."
            )
            sys.exit(1)
        try:
            generator = SanskritImageGenerator()
            image = generator.generate(
                prompt=args.prompt, num_inference_steps=args.steps
            )
            image.save(args.output)
            print(f"Image saved to {args.output}")
        except Exception as e:
            print(f"Image generation failed: {e}")

    elif args.mode == "video":
        print("Initializing Video Generator...")
        if SanskritVideoGenerator is None:
            print(
                "Error: Diffusion models not available. Install 'torch' and 'diffusers'."
            )
            sys.exit(1)
        try:
            if not os.path.exists(args.image):
                print(f"Error: Image not found at {args.image}")
                return

            image = Image.open(args.image).convert("RGB")
            image = image.resize((1024, 576))  # Resize to standard SVD resolution

            generator = SanskritVideoGenerator()
            frames = generator.generate(image)

            # Save frames as video options
            from diffusers.utils import export_to_video

            export_to_video(frames, args.output, fps=7)
            print(f"Video saved to {args.output}")

        except Exception as e:
            print(f"Video generation failed: {e}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
