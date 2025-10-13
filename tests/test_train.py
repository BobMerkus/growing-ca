from unittest.mock import MagicMock, patch

import pytest
import torch

from growing_ca.train import TrainCaModel


class TestTrainCaModel:
    """Test TrainCaModel class."""

    def test_model_initialization_defaults(self):
        """Test model initialization with default values."""
        model = TrainCaModel()
        assert model.epochs == 8000
        assert model.image_path == "data/emojis/emoji_0.png"
        assert model.experiment == "Regenerating"
        assert model.model_path is None
        assert model.lr == 2e-3
        assert model.batch_size == 8
        assert model.pool_size == 1024
        assert model.device == "auto"
        assert model.save_every == 100
        assert model.log_every == 100

    def test_model_initialization_custom(self):
        """Test model initialization with custom values."""
        model = TrainCaModel(
            epochs=1000,
            image_path="custom/image.png",
            experiment="Growing",
            model_path="custom/path.pth",
            lr=1e-3,
            batch_size=16,
            pool_size=512,
            device="cpu",
            save_every=50,
            log_every=25,
        )
        assert model.epochs == 1000
        assert model.image_path == "custom/image.png"
        assert model.experiment == "Growing"
        assert model.model_path == "custom/path.pth"
        assert model.lr == 1e-3
        assert model.batch_size == 16
        assert model.pool_size == 512
        assert model.device == "cpu"
        assert model.save_every == 50
        assert model.log_every == 25

    @patch("growing_ca.train.torch.cuda.is_available")
    @patch("growing_ca.train.torch.cuda.current_device")
    def test_cli_cmd_auto_device_cuda(self, mock_current_device, mock_is_available):
        """Test cli_cmd with auto device when CUDA is available."""
        mock_is_available.return_value = True
        mock_current_device.return_value = 0

        model = TrainCaModel(device="auto", image_path="data/emoji.png")

        with patch("growing_ca.train.Path.exists", return_value=False):
            with pytest.raises(SystemExit) as exc_info:
                model.cli_cmd()
            assert exc_info.value.code == 1

    @patch("growing_ca.train.Path.exists")
    def test_cli_cmd_missing_image_file(self, mock_exists):
        """Test cli_cmd exits when image file is missing."""
        mock_exists.return_value = False
        model = TrainCaModel(device="cpu")

        with pytest.raises(SystemExit) as exc_info:
            model.cli_cmd()
        assert exc_info.value.code == 1

    @patch("growing_ca.train.Path.exists")
    @patch("growing_ca.train.CaTrainer")
    def test_cli_cmd_default_model_path(self, mock_trainer_class, mock_exists):
        """Test cli_cmd uses default model path when not specified."""
        mock_exists.return_value = True
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer

        model = TrainCaModel(
            device="cpu",
            image_path="data/test_image.png",
            epochs=100,
            save_every=10,
            log_every=10,
        )
        model.cli_cmd()

        # Check that trainer was initialized with correct model path (derived from image name)
        assert mock_trainer_class.call_args[1]["model_path"] == "models/test_image.pth"

    @patch("growing_ca.train.Path.exists")
    @patch("growing_ca.train.CaTrainer")
    def test_cli_cmd_custom_model_path(self, mock_trainer_class, mock_exists):
        """Test cli_cmd uses custom model path when specified."""
        mock_exists.return_value = True
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer

        model = TrainCaModel(
            device="cpu",
            model_path="custom/model.pth",
            epochs=100,
            save_every=10,
            log_every=10,
        )
        model.cli_cmd()

        # Check that trainer was initialized with custom model path
        assert mock_trainer_class.call_args[1]["model_path"] == "custom/model.pth"

    @patch("growing_ca.train.Path.exists")
    @patch("growing_ca.train.CaTrainer")
    def test_cli_cmd_successful_training(self, mock_trainer_class, mock_exists):
        """Test successful training flow."""
        mock_exists.return_value = True
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer

        model = TrainCaModel(
            device="cpu",
            epochs=100,
            image_path="data/emojis/emoji_0.png",
            save_every=10,
            log_every=10,
        )
        model.cli_cmd()

        # Check trainer was initialized
        mock_trainer_class.assert_called_once()
        # Check train was called
        mock_trainer.train.assert_called_once_with(
            n_epochs=100, save_every=10, log_every=10
        )

    @patch("growing_ca.train.Path.exists")
    @patch("growing_ca.train.CaTrainer")
    def test_cli_cmd_keyboard_interrupt(self, mock_trainer_class, mock_exists):
        """Test keyboard interrupt during training."""
        mock_exists.return_value = True
        mock_trainer = MagicMock()
        mock_trainer.train.side_effect = KeyboardInterrupt()
        mock_trainer_class.return_value = mock_trainer

        model = TrainCaModel(device="cpu", epochs=100, save_every=10, log_every=10)

        with pytest.raises(SystemExit) as exc_info:
            model.cli_cmd()
        assert exc_info.value.code == 1

    @patch("growing_ca.train.Path.exists")
    @patch("growing_ca.train.CaTrainer")
    def test_cli_cmd_training_error(self, mock_trainer_class, mock_exists):
        """Test error during training."""
        mock_exists.return_value = True
        mock_trainer = MagicMock()
        mock_trainer.train.side_effect = Exception("Training failed")
        mock_trainer_class.return_value = mock_trainer

        model = TrainCaModel(device="cpu", epochs=100, save_every=10, log_every=10)

        with pytest.raises(SystemExit) as exc_info:
            model.cli_cmd()
        assert exc_info.value.code == 1

    @patch("growing_ca.train.Path.exists")
    @patch("growing_ca.train.CaTrainer")
    def test_cli_cmd_trainer_initialization(self, mock_trainer_class, mock_exists):
        """Test that trainer is initialized with correct parameters."""
        mock_exists.return_value = True
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer

        model = TrainCaModel(
            device="cpu",
            image_path="test/image.png",
            experiment="TestExperiment",
            lr=1e-4,
            batch_size=16,
            pool_size=256,
            epochs=50,
            save_every=5,
            log_every=5,
        )
        model.cli_cmd()

        # Verify trainer initialization parameters
        call_kwargs = mock_trainer_class.call_args[1]
        assert call_kwargs["target_image_path"] == "test/image.png"
        assert call_kwargs["experiment_type"] == "TestExperiment"
        assert call_kwargs["device"] == torch.device("cpu")
        assert call_kwargs["lr"] == 1e-4
        assert call_kwargs["batch_size"] == 16
        assert call_kwargs["pool_size"] == 256
