"""Sample test file"""

import pytest
import torch


def test_import():
    """Test that basic imports work"""
    import src
    assert src.__version__ == "0.1.0"


def test_torch_cuda():
    """Test CUDA availability (will skip if no CUDA)"""
    if torch.cuda.is_available():
        assert torch.cuda.device_count() > 0
    else:
        pytest.skip("CUDA not available")


def test_basic_tensor_operations():
    """Test basic PyTorch operations"""
    x = torch.randn(10, 5)
    y = torch.randn(5, 3)
    z = torch.mm(x, y)
    assert z.shape == (10, 3)


# Add more tests for your specific modules
# Example:
# def test_dataset_loading():
#     from src.data.dataset import EchoDataset
#     dataset = EchoDataset(...)
#     assert len(dataset) > 0


