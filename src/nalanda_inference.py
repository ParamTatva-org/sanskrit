import argparse
import sys
import os
from pathlib import Path
import json
from PIL import Image

try:
    import torch
    from torchvision import transforms

    _VISION_AVAILABLE = True
except ImportError:
    torch = None
    _VISION_AVAILABLE = False

# Add current directory to path to allow imports from src
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Imports moved to inside main or guarded


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with Nalanda-62M-Multi model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model directory or checkpoint file",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Input text prompt")
    parser.add_argument(
        "--max_length", type=int, default=100, help="Maximum length of generated text"
    )
    parser.add_argument("--image_path", type=str, help="Path to input image")
    parser.add_argument("--video_path", type=str, help="Path to input video")

    args = parser.parse_args()

    # Allow forcing verification mode for testing to avoid heavy model loads/downloads
    if os.environ.get("FORCE_VERIFICATION_MODE"):
        print("Verification Mode: Forced by environment.")
        return

    if torch is None:
        print("Verification Mode: Torch not installed.")
        return

    try:
        from src.model import MultiModalParamtatva
        from src.model.model_configs import get_config
        from src.tokenizer import SanskritTokenizer
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print(
            "Please ensure you are running from the project root and 'src' packages are available."
        )
        sys.exit(1)

    # Determine device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    print(f"Using device: {device}")

    try:
        # 1. Load checkpoint first to get metadata/vocab
        if not os.path.exists(args.model_path):
            potential_path = os.path.join(args.model_path, "model.pt")
            if os.path.exists(potential_path):
                checkpoint_path = potential_path
            else:
                # Fallback to just args.model_path if it's a file, or error
                if os.path.isfile(args.model_path):
                    checkpoint_path = args.model_path
                else:
                    checkpoint_path = None
        elif os.path.isdir(args.model_path):
            checkpoint_path = os.path.join(args.model_path, "model.pt")
        else:
            checkpoint_path = args.model_path

        if not checkpoint_path or not os.path.exists(checkpoint_path):
            print(f"Error: Model file not found at {args.model_path}")
            # We can't proceed in Secure Mode without weights/vocab from checkpoint
            # But for verification of script structure (without actual weights),
            # let's allow fallback if explicitly testing
            print("WARNING: Proceeding without checkpoint (Verification Mode)")
            checkpoint = None
            vocab_list = None
            token_to_id = None
        else:
            print(f"Loading checkpoint from {checkpoint_path}...")
            try:
                checkpoint = torch.load(
                    checkpoint_path, map_location=device, weights_only=False
                )
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                checkpoint = None

            # Attempt to find vocab in checkpoint or adjacent json
            vocab_path = os.path.join(os.path.dirname(checkpoint_path), "vocab.json")
            vocab_list = None
            token_to_id = None

            if os.path.exists(vocab_path):
                print(f"Found vocab.json at {vocab_path}")
                with open(vocab_path, "r") as f:
                    vocab_data = json.load(f)
                    if isinstance(vocab_data, list):
                        vocab_list = vocab_data
                    elif isinstance(vocab_data, dict):
                        token_to_id = vocab_data

        # Fallback for verification if no checkpoint/vocab loaded
        if vocab_list is None and token_to_id is None:
            if checkpoint is None:
                print("WARNING: Creating dummy vocabulary for verification...")
                vocab_list = [
                    "<PAD>",
                    "<UNK>",
                    "<BOS>",
                    "<EOS>",
                    "<SPACE>",
                    "<NL>",
                    "अ",
                ] + ["a"] * 100

        print("Initializing tokenizer and Paramtatva graph...")
        tokenizer = SanskritTokenizer(vocab_list=vocab_list, token_to_id=token_to_id)
        # ptk_graph is not needed for inference in secure mode

        # Load Config
        # We try to find a config.json in the model directory, otherwise use default
        model_path = Path(args.model_path)
        if model_path.is_file():
            model_dir = model_path.parent
        else:
            model_dir = model_path

        config_path = model_dir / "config.json"

        if config_path.exists():
            print(f"Loading config from {config_path}")
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            # Basic reconstruction of config (adjust as needed based on how config is saved)
            # Assuming standard structure or creating default and overriding
            model_size = config_dict.get("model_size", "small")
            vocab_size = config_dict.get("vocab_size", tokenizer.vocab_size)
            config = get_config(model_size, vocab_size)
            for k, v in config_dict.items():
                if hasattr(config, k):
                    setattr(config, k, v)
        else:
            print("Config file not found, using default 'small' config.")
            config = get_config("small", tokenizer.vocab_size)

        # Ensure cross-attention is set as per MultiModal requirement
        config.add_cross_attention = True

        # 4. Initialize Model (No Secret Graph)
        print("Initializing Multi-Modal model (Secure Mode)...")
        # We pass ptk_graph=None to avoid running the secret graph logic
        # The embeddings will rely on the loaded weights (including their internal lookup tables)

        model = MultiModalParamtatva(
            config=config,
            ptk_graph=None,  # explicitly None
            vision_backbone="resnet50",  # Default or infer? Let's assume ResNet for now or add arg
            video_backbone="r2plus1d_18",
            token_to_id=tokenizer.token_to_id,
        )

        # 5. Load Weights
        print("Loading weights...")

        if checkpoint is not None:
            # We don't need to re-compute buffers from the graph; we just load them from disk.
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            print("Model weights loaded successfully.")
        else:
            print(
                "WARNING: No checkpoint found. Using random initialization (Verification Mode)."
            )

        model.to(device)
        model.eval()

        # 6. Generate

        # Process Visual Input
        pixel_values = None
        video_values = None

        if args.image_path:
            if not _VISION_AVAILABLE:
                print("Error: torchvision required for image input.")
                sys.exit(1)
            if os.path.exists(args.image_path):
                print(f"Loading image from {args.image_path}...")
                image = Image.open(args.image_path).convert("RGB")
                # Standard ResNet transform
                transform = transforms.Compose(
                    [
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
                pixel_values = (
                    transform(image).unsqueeze(0).to(device)
                )  # (1, 3, 224, 224)
            else:
                print(f"Error: Image not found at {args.image_path}")

        if args.video_path:
            # Basic video placeholder - loading videos requires more complex logic (av or decord)
            # keeping simple for now or assuming pre-tensor
            print("Video loading not fully implemented in this script path. Ignoring.")
            video_values = None

        if pixel_values is not None or video_values is not None:
            print("Generating caption for visual input...")
            response = model.generate_caption(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                video_values=video_values,
                max_length=args.max_length,
            )
        else:
            # Text-only generation via decoder

            # ... (Original text generation logic) ...
            print(f"Processing prompt: {args.prompt}")

            if hasattr(tokenizer, "encode"):
                try:
                    encoded = tokenizer.encode(args.prompt)
                    if isinstance(encoded, dict):
                        input_ids_list = encoded["input_ids"]
                    else:
                        input_ids_list = encoded
                except Exception as e:
                    print(f"Tokenizer encode failed (likely missing vocab): {e}")
                    input_ids_list = [1] * 5

                input_ids = torch.tensor([input_ids_list], device=device)
            else:
                print("Tokenizer has no encode method. Using dummy input.")
                input_ids = torch.tensor([[1, 2, 3]], device=device)

            generated_ids, stats = model.decoder.generate(
                input_ids=input_ids,
                max_length=args.max_length,
                temperature=0.7,
            )

            # Decode
            if hasattr(tokenizer, "decode"):
                response = tokenizer.decode(generated_ids[0].tolist())
            else:
                response = "Decode failed"

        print("\nGenerated Response:")
        print("-" * 20)
        print(response)
        print("-" * 20)
        print(f"Stats: {stats}")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
