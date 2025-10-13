import os
from typing import Optional
import logging

import imageio
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from growing_ca.core.model import CAModel
from growing_ca.core.utils_vis import SamplePool, to_rgb, make_seed, make_circle_masks

logger = logging.getLogger(__name__)


class CaTrainer:
    """Trainer for the Cellular Automata model"""

    def __init__(
        self,
        target_image_path: str = "data/emojis/emoji_0.png",
        model_path: str = "models/emoji_0.pth",
        experiment_type: str = "Regenerating",
        device: Optional[torch.device] = None,
        channel_n: int = 16,
        target_padding: int = 16,
        lr: float = 2e-3,
        lr_gamma: float = 0.9999,
        betas: tuple[float, float] = (0.5, 0.5),
        batch_size: int = 8,
        pool_size: int = 1024,
        cell_fire_rate: float = 0.5,
        hidden_size: int = 128,
    ):
        self.target_image_path = target_image_path
        self.model_path = model_path
        self.channel_n = channel_n
        self.target_padding = target_padding
        self.lr = lr
        self.lr_gamma = lr_gamma
        self.betas = betas
        self.batch_size = batch_size
        self.pool_size = pool_size
        self.cell_fire_rate = cell_fire_rate
        self.hidden_size = hidden_size

        if device is None:
            device = torch.device(torch.cuda.current_device())
        else:
            self.device = device

        # Experiment configuration
        experiment_map = {"Growing": 0, "Persistent": 1, "Regenerating": 2}
        if experiment_type not in experiment_map:
            raise ValueError(f"Unknown experiment type: {experiment_type}")

        experiment_n = experiment_map[experiment_type]
        self.use_pattern_pool = [False, True, True][experiment_n]
        self.damage_n = [0, 0, 3][experiment_n]

        self.loss_log: list[float] = []

        # Initialize components
        self._setup_target()
        self._setup_model()
        self._setup_optimizer()
        self._setup_pool()

    @staticmethod
    def load_image(path: str) -> np.ndarray:
        """Load an image from a path and ensure it has 4 channels (RGBA)"""
        im = imageio.imread(path)
        img = np.array(im.astype(np.float32))
        img /= 255.0

        # Ensure image has 4 channels (RGBA)
        if not img.shape[-1] == 4:
            raise ValueError(f"Image must have 4 (RGBA) channels, got {img.shape[-1]}")

        return img

    def _setup_target(self) -> None:
        """Setup the target image"""
        target_img = self.load_image(self.target_image_path)
        p = self.target_padding
        pad_target = np.pad(target_img, [(p, p), (p, p), (0, 0)])
        self.h, self.w = pad_target.shape[:2]
        pad_target = np.expand_dims(pad_target, axis=0)
        self.pad_target = torch.from_numpy(pad_target.astype(np.float32)).to(
            self.device
        )

    def _setup_model(self) -> None:
        """Setup the model"""
        self.ca = CAModel(
            self.channel_n,
            self.cell_fire_rate,
            self.device,
            hidden_size=self.hidden_size,
        ).to(self.device)

        # Load existing model if it exists
        if os.path.exists(self.model_path):
            self.ca.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
            logger.info(f"Loaded existing model from {self.model_path}")
        else:
            logger.info(f"Starting with new model - will save to {self.model_path}")

    def _setup_optimizer(self) -> None:
        """Setup the optimizer"""
        self.optimizer = optim.Adam(self.ca.parameters(), lr=self.lr, betas=self.betas)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, self.lr_gamma)

    def _setup_pool(self) -> None:
        """Setup the pool"""
        self.seed = make_seed((self.h, self.w), self.channel_n)
        self.pool: SamplePool | None
        if self.use_pattern_pool:
            self.pool = SamplePool(x=np.repeat(self.seed[None, ...], self.pool_size, 0))
        else:
            self.pool = None

    def loss_f(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Loss function"""
        return torch.mean(torch.pow(x[..., :4] - target, 2), [-2, -3, -1])

    def train_step(
        self, x: torch.Tensor, target: torch.Tensor, steps: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Train a step"""
        x = self.ca(x, steps=steps)
        loss = F.mse_loss(x[:, :, :, :4], target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return x, loss

    def get_batch(self) -> tuple[torch.Tensor, SamplePool | None]:
        """Get a batch"""
        if self.use_pattern_pool and self.pool is not None:
            batch = self.pool.sample(self.batch_size)
            batch_x: np.ndarray = getattr(batch, "x")
            x0_tensor = torch.from_numpy(batch_x.astype(np.float32)).to(self.device)
            loss_rank_array: np.ndarray = (
                self.loss_f(x0_tensor, self.pad_target)
                .detach()
                .cpu()
                .numpy()
                .argsort()[::-1]
            )
            x0 = batch_x[loss_rank_array]
            x0[:1] = self.seed
            if self.damage_n:
                damage = (
                    1.0 - make_circle_masks(self.damage_n, self.h, self.w)[..., None]
                )
                x0[-self.damage_n :] *= damage
            return torch.from_numpy(x0.astype(np.float32)).to(self.device), batch
        else:
            x0_np: np.ndarray = np.repeat(self.seed[None, ...], self.batch_size, 0)
            return torch.from_numpy(x0_np.astype(np.float32)).to(self.device), None

    def train(
        self, n_epochs: int = 8000, save_every: int = 100, log_every: int = 100
    ) -> None:
        logger.info(f"Starting training for {n_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(
            f"Experiment mode: {'Pool-based' if self.use_pattern_pool else 'Seed-based'}"
        )
        logger.info(f"Damage patterns: {self.damage_n}")

        for i in range(n_epochs + 1):
            x0, batch = self.get_batch()

            # Random number of steps between 64-96 as in original
            steps = np.random.randint(64, 96)
            x, loss = self.train_step(x0, self.pad_target, steps)

            if self.use_pattern_pool and batch is not None:
                batch_x_attr: np.ndarray = getattr(batch, "x")
                batch_x_attr[:] = x.detach().cpu().numpy()
                batch.commit()

            self.loss_log.append(loss.item())

            if i % log_every == 0:
                logger.info(
                    f"Epoch {i:6d}, Loss: {loss.item():.6f}, LR: {self.scheduler.get_last_lr()[0]:.2e}"
                )

            if i % save_every == 0 and i > 0:
                self.save_model()

        self.save_model()
        logger.info("Training completed!")

    def save_model(self) -> None:
        """Save the model"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.ca.state_dict(), self.model_path)
        logger.info(f"Model saved to {self.model_path}")

    def get_loss_history(self) -> list[float]:
        """Get the loss history"""
        return self.loss_log.copy()

    def visualize_current_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Visualize the current state"""
        x0, _ = self.get_batch()
        with torch.no_grad():
            x = self.ca(x0, steps=64)

        vis0 = to_rgb(x0.detach().cpu().numpy())
        vis1 = to_rgb(x.detach().cpu().numpy())
        return vis0, vis1
