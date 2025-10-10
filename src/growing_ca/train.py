#!/usr/bin/env python3
from pydantic import BaseModel, Field
import sys
from pathlib import Path

import torch

from growing_ca.core.trainer import CaTrainer


class TrainCaModel(BaseModel):
    """Train a Cellular Automata model for a given emoji"""

    epochs: int = Field(default=8000, description="Number of epochs to train for")
    emoji_index: int = Field(default=0, description="Index of the emoji to train on")
    experiment: str = Field(default="Regenerating", description="Experiment type")

    model_path: str | None = Field(
        default=None,
        description="Path to save/load the model (defaults to models/emoji_{emoji_index}.pth)",
    )
    emoji_path: str = Field(
        default="data/emoji.png", description="Path to the emoji to train on"
    )
    lr: float = Field(default=2e-3, description="Learning rate")
    batch_size: int = Field(default=8, description="Batch size")
    pool_size: int = Field(default=1024, description="Size of the pool to use")

    device: str = Field(default="auto", description="Device to use")
    save_every: int = Field(default=100, description="Save every n epochs")
    log_every: int = Field(default=100, description="Log every n epochs")

    def cli_cmd(self) -> None:
        # Set device
        if self.device == "auto":
            device = torch.device(torch.cuda.current_device())
        else:
            device = torch.device(self.device)

        # Determine model path based on emoji index if not specified
        model_path = self.model_path
        if model_path is None:
            model_path = f"models/emoji_{self.emoji_index}.pth"

        # Check if emoji file exists
        if not Path(self.emoji_path).exists():
            print(f"Error: Emoji file not found at {self.emoji_path}")
            print(
                "Please ensure the emoji dataset is available or specify the correct path with --emoji-path"
            )
            sys.exit(1)

        print("=" * 60)
        print("Growing Neural Cellular Automata Training")
        print("=" * 60)
        print(f"Emoji index: {self.emoji_index}")
        print(f"Experiment type: {self.experiment}")
        print(f"Epochs: {self.epochs}")
        print(f"Device: {device}")
        print(f"Model path: {model_path}")
        print("=" * 60)

        try:
            # Initialize trainer
            trainer = CaTrainer(
                target_emoji_index=self.emoji_index,
                emoji_path=self.emoji_path,
                model_path=model_path,
                experiment_type=self.experiment,
                device=device,
                lr=self.lr,
                batch_size=self.batch_size,
                pool_size=self.pool_size,
            )

            # Start training
            trainer.train(
                n_epochs=self.epochs,
                save_every=self.save_every,
                log_every=self.log_every,
            )

            print("\nTraining completed successfully!")
            print(f"Final model saved to: {model_path}")

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\nError during training: {e}")
            sys.exit(1)
