#!/usr/bin/env python3
from pydantic import BaseModel, Field
import sys
from pathlib import Path

import torch

from growing_ca.core.trainer import CaTrainer

import logging


class TrainCaModel(BaseModel):
    """Train a Cellular Automata model for a given target image"""

    epochs: int = Field(default=8000, description="Number of epochs to train for")
    image_path: str = Field(
        default="data/emojis/emoji_0.png",
        description="Path to the target image to train on",
    )
    experiment: str = Field(default="Regenerating", description="Experiment type")
    model_path: str | None = Field(
        default=None,
        description="Path to save/load the model (defaults to models/{image_name}.pth)",
    )
    lr: float = Field(default=2e-3, description="Learning rate")
    batch_size: int = Field(default=8, description="Batch size")
    pool_size: int = Field(default=1024, description="Size of the pool to use")
    cell_fire_rate: float = Field(default=0.5, description="Cell fire rate")
    hidden_size: int = Field(default=128, description="Hidden size")
    device: str = Field(default="auto", description="Device to use")
    save_every: int = Field(default=100, description="Save every n epochs")
    log_every: int = Field(default=100, description="Log every n epochs")

    def cli_cmd(self) -> None:
        # Set device
        logging.basicConfig(level=logging.INFO)
        if self.device == "auto":
            device = torch.device(torch.cuda.current_device())
        else:
            device = torch.device(self.device)

        # Check if image file exists
        if not Path(self.image_path).exists():
            logging.error(f"Error: Image file not found at {self.image_path}")
            logging.error(
                "Please ensure the image file is available or specify the correct path with --image-path"
            )
            sys.exit(1)

        # Determine model path based on image name if not specified
        model_path = self.model_path
        if model_path is None:
            image_name = Path(self.image_path).stem
            model_path = f"models/{image_name}.pth"

        logging.info("=" * 60)
        logging.info("Growing Neural Cellular Automata Training")
        logging.info("=" * 60)
        logging.info(f"Target image: {self.image_path}")
        logging.info(f"Experiment type: {self.experiment}")
        logging.info(f"Epochs: {self.epochs}")
        logging.info(f"Device: {device}")
        logging.info(f"Model path: {model_path}")
        logging.info("=" * 60)

        try:
            # Initialize trainer
            trainer = CaTrainer(
                target_image_path=self.image_path,
                model_path=model_path,
                experiment_type=self.experiment,
                device=device,
                lr=self.lr,
                batch_size=self.batch_size,
                pool_size=self.pool_size,
                cell_fire_rate=self.cell_fire_rate,
                hidden_size=self.hidden_size,
            )

            # Start training
            trainer.train(
                n_epochs=self.epochs,
                save_every=self.save_every,
                log_every=self.log_every,
            )

            logging.info("\nTraining completed successfully!")
            logging.info(f"Final model saved to: {model_path}")

        except KeyboardInterrupt:
            logging.info("\nTraining interrupted by user")
            sys.exit(1)
        except Exception as e:
            logging.exception(f"\nError during training: {e}")
            sys.exit(1)
