import unittest
import sys
import os
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestCPUOptimization(unittest.TestCase):
    def setUp(self):
        self.nalanda_script = "src.nalanda_inference"
        self.diffusion_script = "diffusion_inference"

        # Mock modules that might be missing in the test environment
        modules_to_patch = {
            "torch": MagicMock(),
            "torch.nn": MagicMock(),
            "torch.cuda": MagicMock(),
            "torch.backends": MagicMock(),
            "torch.backends.mps": MagicMock(),
            "torch.quantization": MagicMock(),
            "diffusers": MagicMock(),
            "diffusers.utils": MagicMock(),
            "torchvision": MagicMock(),
            "torchvision.models": MagicMock(),
            "torchvision.models.video": MagicMock(),
        }
        self.torch_patcher = patch.dict(sys.modules, modules_to_patch)
        self.torch_patcher.start()

        # Mock sys.argv to allow importing main modules without triggering argparse
        self.argv_patcher = patch.object(sys, "argv", ["prog"])
        self.argv_patcher.start()

    def tearDown(self):
        self.torch_patcher.stop()
        self.argv_patcher.stop()

    @patch("src.nalanda_inference.torch")
    @patch("src.model.MultiModalParamtatva")
    @patch("src.tokenizer.SanskritTokenizer")
    @patch("src.model.model_configs.get_config")
    def test_nalanda_cpu_quantization(
        self, mock_config, mock_tokenizer, mock_model_cls, mock_torch
    ):
        """Test that nalanda_inference applies quantization when requested."""
        from src.nalanda_inference import main

        # Setup mocks
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        mock_model_instance = MagicMock()
        mock_model_cls.return_value = mock_model_instance

        # Mock generate return value (ids, stats)
        mock_model_instance.decoder.generate.return_value = (MagicMock(), {})

        # Mock quantization
        mock_torch.quantization.quantize_dynamic = MagicMock()
        mock_torch.quantization.quantize_dynamic.return_value = mock_model_instance

        # Run main with cpu and quantization args
        test_args = [
            "nalanda_inference.py",
            "--model_path",
            "dummy_path",
            "--prompt",
            "test prompt",
            "--device",
            "cpu",
            "--cpu_quantize",
        ]

        with patch.object(sys, "argv", test_args):
            # We also need to mock os.path.exists to avoid early exits or loading real files
            with patch("src.nalanda_inference.os.path.exists") as mock_exists:
                # Let it think model_path doesn't exist so it enters verification mode (no checkpoint)
                mock_exists.return_value = False

                # Ensure FORCE_VERIFICATION_MODE is not set
                with patch.dict(os.environ, {}, clear=True):
                    main()

        # Check if quantize_dynamic was called
        # Note: The script only calls quantize_dynamic if device is cpu and args.cpu_quantize is True
        # and model is initialized.
        mock_torch.quantization.quantize_dynamic.assert_called_once()
        print("\nVerified: nalanda_inference called quantize_dynamic.")

    @patch("diffusion_inference.SanskritImageGenerator")
    @patch("diffusion_inference.SanskritVideoGenerator")
    @patch("diffusion_inference.torch")
    def test_diffusion_device_passing(self, mock_torch, mock_video_gen, mock_image_gen):
        """Test that diffusion_inference passes device to generators."""
        from diffusion_inference import main

        # Case 1: Image with CPU
        test_args_img = [
            "diffusion_inference.py",
            "image",
            "--prompt",
            "test",
            "--device",
            "cpu",
        ]

        mock_image_gen_instance = MagicMock()
        mock_image_gen.return_value = mock_image_gen_instance

        with patch.object(sys, "argv", test_args_img):
            main()

        mock_image_gen.assert_called_with(device="cpu")
        print("\nVerified: SanskritImageGenerator initialized with device='cpu'.")

        # Case 2: Video with CUDA
        test_args_vid = [
            "diffusion_inference.py",
            "video",
            "--image",
            "dummy.jpg",
            "--device",
            "cuda",
        ]

        with patch.object(sys, "argv", test_args_vid):
            with patch("diffusion_inference.os.path.exists") as mock_exists:
                mock_exists.return_value = True
                with patch("diffusion_inference.Image.open"):
                    main()

        mock_video_gen.assert_called_with(device="cuda")
        print("\nVerified: SanskritVideoGenerator initialized with device='cuda'.")


if __name__ == "__main__":
    unittest.main()
