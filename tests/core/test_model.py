import pytest
import torch

from growing_ca.core.model import CAModel


class TestCAModel:
    """Test CAModel class."""

    @pytest.fixture
    def device(self):
        """Get available device."""
        return "cpu"  # Always use CPU for tests

    @pytest.fixture
    def model(self, device):
        """Create a test model."""
        return CAModel(channel_n=16, fire_rate=0.5, device=torch.device(device))

    def test_model_initialization(self, device):
        """Test model initialization."""
        model = CAModel(channel_n=16, fire_rate=0.5, device=torch.device(device))
        assert model is not None
        assert model.channel_n == 16
        assert model.fire_rate == 0.5
        assert model.device == torch.device(device)

    def test_model_with_custom_hidden_size(self, device):
        """Test model with custom hidden size."""
        model = CAModel(
            channel_n=16, fire_rate=0.5, device=torch.device(device), hidden_size=256
        )
        assert model.fc0.out_features == 256

    def test_alive_function(self, model):
        """Test alive function."""
        # Create input with alive cells (alpha > 0.1)
        x = torch.zeros(1, 16, 10, 10).to(model.device)
        x[:, 3:4, 5, 5] = 0.5  # Set alpha channel
        alive_mask = model.alive(x)
        assert alive_mask.shape == (1, 1, 10, 10)
        assert alive_mask.dtype == torch.bool

    def test_perceive_function(self, model):
        """Test perceive function."""
        x = torch.randn(1, 16, 10, 10).to(model.device)
        perceived = model.perceive(x, angle=0.0)
        # Should concatenate original + 2 convolutions = 3 * 16 = 48 channels
        assert perceived.shape == (1, 48, 10, 10)

    def test_perceive_with_angle(self, model):
        """Test perceive function with different angles."""
        x = torch.randn(1, 16, 10, 10).to(model.device)
        perceived_0 = model.perceive(x, angle=0.0)
        perceived_90 = model.perceive(x, angle=90.0)
        # Different angles should produce different results
        assert not torch.allclose(perceived_0, perceived_90)

    def test_update_function(self, model):
        """Test update function."""
        x = torch.randn(1, 10, 10, 16).to(model.device)
        x[:, 5, 5, 3:] = 1.0  # Set alive cell
        updated = model.update(x, fire_rate=0.5, angle=0.0)
        assert updated.shape == x.shape

    def test_update_with_none_fire_rate(self, model):
        """Test update with None fire_rate uses default."""
        x = torch.randn(1, 10, 10, 16).to(model.device)
        x[:, 5, 5, 3:] = 1.0
        updated = model.update(x, fire_rate=None, angle=0.0)
        assert updated.shape == x.shape

    def test_forward_single_step(self, model):
        """Test forward pass with single step."""
        x = torch.randn(1, 10, 10, 16).to(model.device)
        x[:, 5, 5, 3:] = 1.0
        output = model.forward(x, steps=1)
        assert output.shape == x.shape

    def test_forward_multiple_steps(self, model):
        """Test forward pass with multiple steps."""
        x = torch.randn(1, 10, 10, 16).to(model.device)
        x[:, 5, 5, 3:] = 1.0
        output = model.forward(x, steps=5)
        assert output.shape == x.shape

    def test_forward_with_custom_fire_rate(self, model):
        """Test forward pass with custom fire rate."""
        x = torch.randn(1, 10, 10, 16).to(model.device)
        x[:, 5, 5, 3:] = 1.0
        output = model.forward(x, steps=1, fire_rate=0.8)
        assert output.shape == x.shape

    def test_forward_with_angle(self, model):
        """Test forward pass with custom angle."""
        x = torch.randn(1, 10, 10, 16).to(model.device)
        x[:, 5, 5, 3:] = 1.0
        output = model.forward(x, steps=1, angle=45.0)
        assert output.shape == x.shape

    def test_fc1_weight_initialized_to_zero(self, model):
        """Test that fc1 weights are initialized to zero."""
        assert torch.allclose(model.fc1.weight, torch.zeros_like(model.fc1.weight))
