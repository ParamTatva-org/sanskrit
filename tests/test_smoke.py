import sys
import os
from unittest.mock import MagicMock, patch

# Add src to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))


def test_imports():
    """Simple smoke test to check if modules can be imported."""
    # Mock torch and other heavy dependencies
    modules_to_mock = {
        "torch": MagicMock(),
        "torch.nn": MagicMock(),
        "torch.nn.functional": MagicMock(),
        "transformers": MagicMock(),
        "PIL": MagicMock(),
        "torchvision": MagicMock(),
        "torchvision.models": MagicMock(),
        "torchvision.models.video": MagicMock(),
    }

    # Ensure all mocks have __spec__ to satisfy importlib checks
    for m in modules_to_mock.values():
        m.__spec__ = MagicMock()

    with patch.dict(sys.modules, modules_to_mock):
        try:
            # We need to invalidate the cache for nalanda_inference if it was already imported?
            # Or just import it. If it was already imported by another test, this test usually
            # checks if it *can* be imported.
            # If we want to test import logic specifically with mocks, we should probably
            # reload it or remove it from sys.modules first.
            if "nalanda_inference" in sys.modules:
                del sys.modules["nalanda_inference"]
            if "src.nalanda_inference" in sys.modules:
                del sys.modules["src.nalanda_inference"]

            import nalanda_inference

            # Ensure it's strictly the module we expect
            assert nalanda_inference is not None
            assert True
        except ImportError as e:
            assert False, f"Failed to import nalanda_inference: {e}"
