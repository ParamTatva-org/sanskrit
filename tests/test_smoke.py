import sys
import os
from unittest.mock import MagicMock

# Add src to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

def test_imports():
    """Simple smoke test to check if modules can be imported."""
    # Mock torch and other heavy dependencies
    sys.modules["torch"] = MagicMock()
    sys.modules["torch.nn"] = MagicMock()
    sys.modules["torch.nn.functional"] = MagicMock()
    sys.modules["transformers"] = MagicMock()
    sys.modules["PIL"] = MagicMock()
    sys.modules["torchvision"] = MagicMock()
    sys.modules["torchvision.models"] = MagicMock()
    sys.modules["torchvision.models.video"] = MagicMock()
    
    try:
        import nalanda_inference
        assert True
    except ImportError as e:
        assert False, f"Failed to import nalanda_inference: {e}"
