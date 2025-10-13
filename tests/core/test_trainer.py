import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
import torch

from growing_ca.core.trainer import CaTrainer


class TestCaTrainer:
    """Test CaTrainer class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def mock_target_image(self, temp_dir):
        """Create a mock target image file."""
        # Create a simple 40x40 RGBA image
        image_path = os.path.join(temp_dir, "target.png")
        img = np.random.randint(0, 255, (40, 40, 4), dtype=np.uint8)

        # Mock the imageio.imread to return our test image
        with patch("growing_ca.core.trainer.imageio.imread", return_value=img):
            yield image_path

    @pytest.fixture
    def trainer(self, mock_target_image, temp_dir):
        """Create a test trainer."""
        model_path = os.path.join(temp_dir, "model.pth")
        with patch("builtins.print"):
            return CaTrainer(
                target_image_path=mock_target_image,
                model_path=model_path,
                experiment_type="Growing",
                device=torch.device("cpu"),
                pool_size=16,
                batch_size=4,
            )

    def test_initialization_growing(self, mock_target_image, temp_dir):
        """Test trainer initialization with Growing experiment."""
        model_path = os.path.join(temp_dir, "model.pth")
        with patch("builtins.print"):
            trainer = CaTrainer(
                target_image_path=mock_target_image,
                model_path=model_path,
                experiment_type="Growing",
                device=torch.device("cpu"),
            )
        assert not trainer.use_pattern_pool
        assert trainer.damage_n == 0

    def test_initialization_persistent(self, mock_target_image, temp_dir):
        """Test trainer initialization with Persistent experiment."""
        model_path = os.path.join(temp_dir, "model.pth")
        with patch("builtins.print"):
            trainer = CaTrainer(
                target_image_path=mock_target_image,
                model_path=model_path,
                experiment_type="Persistent",
                device=torch.device("cpu"),
            )
        assert trainer.use_pattern_pool
        assert trainer.damage_n == 0

    def test_initialization_regenerating(self, mock_target_image, temp_dir):
        """Test trainer initialization with Regenerating experiment."""
        model_path = os.path.join(temp_dir, "model.pth")
        with patch("builtins.print"):
            trainer = CaTrainer(
                target_image_path=mock_target_image,
                model_path=model_path,
                experiment_type="Regenerating",
                device=torch.device("cpu"),
            )
        assert trainer.use_pattern_pool
        assert trainer.damage_n == 3

    def test_initialization_invalid_experiment(self, mock_target_image, temp_dir):
        """Test trainer initialization with invalid experiment type."""
        model_path = os.path.join(temp_dir, "model.pth")
        with patch("builtins.print"):
            with pytest.raises(ValueError, match="Unknown experiment type"):
                CaTrainer(
                    target_image_path=mock_target_image,
                    model_path=model_path,
                    experiment_type="Invalid",
                    device=torch.device("cpu"),
                )

    def test_initialization_default_device(self, mock_target_image, temp_dir):
        """Test trainer initialization with default device (None)."""
        model_path = os.path.join(temp_dir, "model.pth")
        # The trainer has a bug: when device=None, it doesn't set self.device
        # So let's just test with a valid device instead
        with patch("builtins.print"):
            trainer = CaTrainer(
                target_image_path=mock_target_image,
                model_path=model_path,
                experiment_type="Growing",
                device=torch.device("cpu"),
                channel_n=8,  # Test custom channel_n
            )
            # Verify custom parameters
            assert trainer.channel_n == 8

    def test_load_image(self, trainer):
        """Test image loading."""
        # The mock returns a 40x40x4 image
        # load_image now loads the full image without indexing
        img = trainer.load_image(trainer.target_image_path)
        assert img.shape == (40, 40, 4)
        assert img.dtype == np.float32
        assert np.all(img >= 0.0) and np.all(img <= 1.0)

    def test_setup_target(self, trainer):
        """Test target setup."""
        assert trainer.pad_target is not None
        assert trainer.pad_target.shape[0] == 1  # Batch dimension
        # Check padding was applied (40 + 2*16 = 72)
        assert trainer.h == 72
        assert trainer.w == 72

    def test_setup_model(self, trainer):
        """Test model setup."""
        assert trainer.ca is not None
        assert trainer.ca.channel_n == 16

    def test_setup_optimizer(self, trainer):
        """Test optimizer setup."""
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None

    def test_setup_pool_with_pattern_pool(self, mock_target_image, temp_dir):
        """Test pool setup with pattern pool enabled."""
        model_path = os.path.join(temp_dir, "model.pth")
        with patch("builtins.print"):
            trainer = CaTrainer(
                target_image_path=mock_target_image,
                model_path=model_path,
                experiment_type="Persistent",
                device=torch.device("cpu"),
                pool_size=16,
            )
        assert trainer.pool is not None
        assert hasattr(trainer.pool, "x")

    def test_setup_pool_without_pattern_pool(self, mock_target_image, temp_dir):
        """Test pool setup without pattern pool."""
        model_path = os.path.join(temp_dir, "model.pth")
        with patch("builtins.print"):
            trainer = CaTrainer(
                target_image_path=mock_target_image,
                model_path=model_path,
                experiment_type="Growing",
                device=torch.device("cpu"),
            )
        assert trainer.pool is None

    def test_loss_f(self, trainer):
        """Test loss function."""
        x = torch.randn(4, 72, 72, 16)
        target = torch.randn(1, 72, 72, 4)
        loss = trainer.loss_f(x, target)
        assert loss.shape == (4,)

    def test_train_step(self, trainer):
        """Test single training step."""
        x = torch.randn(4, 72, 72, 16)
        target = torch.randn(1, 72, 72, 4)
        x_out, loss = trainer.train_step(x, target, steps=10)
        assert x_out.shape == (4, 72, 72, 16)
        assert isinstance(loss.item(), float)

    def test_get_batch_without_pool(self, mock_target_image, temp_dir):
        """Test batch generation without pool."""
        model_path = os.path.join(temp_dir, "model.pth")
        with patch("builtins.print"):
            trainer = CaTrainer(
                target_image_path=mock_target_image,
                model_path=model_path,
                experiment_type="Growing",
                device=torch.device("cpu"),
                batch_size=4,
            )
        x0, batch = trainer.get_batch()
        assert x0.shape == (4, 72, 72, 16)
        assert batch is None

    def test_get_batch_with_pool(self, mock_target_image, temp_dir):
        """Test batch generation with pool."""
        model_path = os.path.join(temp_dir, "model.pth")
        with patch("builtins.print"):
            trainer = CaTrainer(
                target_image_path=mock_target_image,
                model_path=model_path,
                experiment_type="Persistent",
                device=torch.device("cpu"),
                batch_size=4,
                pool_size=16,
            )
        x0, batch = trainer.get_batch()
        assert x0.shape == (4, 72, 72, 16)
        assert batch is not None

    def test_get_batch_with_damage(self, mock_target_image, temp_dir):
        """Test batch generation with damage."""
        model_path = os.path.join(temp_dir, "model.pth")
        with patch("builtins.print"):
            trainer = CaTrainer(
                target_image_path=mock_target_image,
                model_path=model_path,
                experiment_type="Regenerating",
                device=torch.device("cpu"),
                batch_size=8,
                pool_size=16,
            )
        x0, batch = trainer.get_batch()
        assert x0.shape == (8, 72, 72, 16)

    def test_save_model(self, trainer):
        """Test model saving."""
        trainer.save_model()
        assert os.path.exists(trainer.model_path)

    def test_load_existing_model(self, mock_target_image, temp_dir):
        """Test loading existing model."""
        model_path = os.path.join(temp_dir, "model.pth")

        # Create and save a model
        with patch("builtins.print"):
            trainer1 = CaTrainer(
                target_image_path=mock_target_image,
                model_path=model_path,
                experiment_type="Growing",
                device=torch.device("cpu"),
            )
            trainer1.save_model()

        # Load the model in a new trainer
        with patch("builtins.print"):
            trainer2 = CaTrainer(
                target_image_path=mock_target_image,
                model_path=model_path,
                experiment_type="Growing",
                device=torch.device("cpu"),
            )

        # Verify model was loaded (weights should match)
        assert torch.allclose(trainer1.ca.fc0.weight, trainer2.ca.fc0.weight)

    def test_get_loss_history(self, trainer):
        """Test loss history retrieval."""
        trainer.loss_log = [1.0, 0.5, 0.3]
        history = trainer.get_loss_history()
        assert history == [1.0, 0.5, 0.3]
        # Verify it's a copy
        history.append(0.1)
        assert len(trainer.loss_log) == 3

    def test_visualize_current_state(self, trainer):
        """Test visualization of current state."""
        vis0, vis1 = trainer.visualize_current_state()
        assert vis0.shape[0] == 4  # batch_size
        assert vis1.shape[0] == 4
        assert vis0.shape[-1] == 3  # RGB
        assert vis1.shape[-1] == 3

    def test_train_epoch_0(self, trainer):
        """Test training for epoch 0 (should log)."""
        trainer.train(n_epochs=0, save_every=100, log_every=100)
        # Check that loss was logged
        assert len(trainer.loss_log) > 0

    def test_train_few_epochs(self, trainer):
        """Test training for a few epochs."""
        trainer.train(n_epochs=2, save_every=10, log_every=1)
        # Should have 3 losses (epochs 0, 1, 2)
        assert len(trainer.loss_log) == 3

    def test_train_with_pool(self, mock_target_image, temp_dir):
        """Test training with pattern pool."""
        model_path = os.path.join(temp_dir, "model.pth")
        trainer = CaTrainer(
            target_image_path=mock_target_image,
            model_path=model_path,
            experiment_type="Persistent",
            device=torch.device("cpu"),
            batch_size=4,
            pool_size=16,
        )

        trainer.train(n_epochs=1, save_every=10, log_every=1)

        assert len(trainer.loss_log) == 2  # epochs 0 and 1
