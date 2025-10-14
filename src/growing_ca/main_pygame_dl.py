from pydantic import BaseModel, Field
import torch

from growing_ca.core.displayer import displayer
from growing_ca.core.utils import mat_distance
from growing_ca.core.model import CAModel
from growing_ca.core.utils_vis import to_rgb, make_seed
import numpy as np
import logging

logger = logging.getLogger(__name__)


def visualize(
    eraser_radius: int,
    pix_size: int,
    map_shape: tuple[int, int],
    channel_n: int,
    cell_fire_rate: float,
    model_path: str,
    device: torch.device,
    image_path: str,
) -> None:
    import pygame
    import imageio

    # Load target image for visualization
    def load_image(path: str) -> np.ndarray:
        im = imageio.imread(path)
        img = np.array(im.astype(np.float32))
        img /= 255.0
        return img

    target_image = load_image(image_path)

    _rows: np.ndarray = (
        np.arange(map_shape[0])
        .repeat(map_shape[1])
        .reshape([map_shape[0], map_shape[1]])
    )
    _cols: np.ndarray = (
        np.arange(map_shape[1]).reshape([1, -1]).repeat(map_shape[0], axis=0)
    )
    _map_pos: np.ndarray = np.array([_rows, _cols]).transpose([1, 2, 0])

    _map: np.ndarray = make_seed(map_shape, channel_n)

    model: CAModel = CAModel(channel_n, cell_fire_rate, device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    output: torch.Tensor = model(
        torch.from_numpy(
            _map.reshape([1, map_shape[0], map_shape[1], channel_n]).astype(np.float32)
        ).to(device),
        1,
    )

    # Create a window that shows both the CA and the target image side by side
    disp: displayer = displayer(map_shape, pix_size, target_image=target_image)

    isMouseDown: bool = False
    running: bool = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    isMouseDown = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    isMouseDown = False

        if isMouseDown:
            try:
                mouse_pos: np.ndarray = np.array(
                    [int(event.pos[1] / pix_size), int(event.pos[0] / pix_size)]
                )
                should_keep: np.ndarray = (
                    mat_distance(_map_pos, mouse_pos) > eraser_radius
                ).reshape([map_shape[0], map_shape[1], 1])
                output = torch.from_numpy(output.cpu().detach().numpy() * should_keep)
            except AttributeError:
                pass

        output = model(output, 1)

        _map = to_rgb(output.cpu().detach().numpy()[0])
        disp.update(_map)


class VisualizeCaModel(BaseModel):
    """Visualize a Cellular Automata model for a given target image"""

    eraser_radius: int = Field(default=3, description="Eraser radius, in pixels")
    pix_size: int = Field(default=8, description="Pixel size, in pixels")
    map_shape: tuple[int, int] = Field(
        default=(72, 72), description="Map shape, in pixels"
    )
    channel_n: int = Field(default=16, description="Number of channels")
    cell_fire_rate: float = Field(default=0.5, description="Cell fire rate")
    model_path: str | None = Field(
        default=None,
        description="Model path (defaults to models/{image_name}.pth)",
    )
    device: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu",
        description="Device",
    )
    image_path: str = Field(
        default="data/emojis/emoji_0.png", description="Path to the target image"
    )

    def cli_cmd(self) -> None:
        from pathlib import Path

        logging.basicConfig(level=logging.INFO)
        logger.info(f"Visualizing model at {self.model_dump()}")

        # Determine model path based on image name if not specified
        model_path = self.model_path
        if model_path is None:
            image_name = Path(self.image_path).stem
            model_path = f"models/{image_name}.pth"
            logger.info(f"Using model path: {model_path}")

        visualize(
            eraser_radius=self.eraser_radius,
            pix_size=self.pix_size,
            map_shape=self.map_shape,
            channel_n=self.channel_n,
            cell_fire_rate=self.cell_fire_rate,
            model_path=model_path,
            device=torch.device(self.device),
            image_path=self.image_path,
        )


if __name__ == "__main__":
    visualize(
        eraser_radius=3,
        pix_size=8,
        map_shape=(72, 72),
        channel_n=16,
        cell_fire_rate=0.5,
        model_path="models/emoji_0.pth",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        image_path="data/emojis/emoji_0.png",
    )
