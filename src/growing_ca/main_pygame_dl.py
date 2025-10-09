import os
import pygame
import torch
import numpy as np

from growing_ca.core.displayer import displayer
from growing_ca.core.utils import mat_distance
from growing_ca.core.CAModel import CAModel
from growing_ca.core.utils_vis import to_rgb, make_seed

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def main() -> None:
    eraser_radius: int = 3
    pix_size: int = 8
    _map_shape: tuple[int, int] = (72, 72)
    CHANNEL_N: int = 16
    CELL_FIRE_RATE: float = 0.5
    model_path: str = "models/remaster_1.pth"
    device: torch.device = torch.device("cpu")

    _rows: np.ndarray = (
        np.arange(_map_shape[0])
        .repeat(_map_shape[1])
        .reshape([_map_shape[0], _map_shape[1]])
    )
    _cols: np.ndarray = (
        np.arange(_map_shape[1]).reshape([1, -1]).repeat(_map_shape[0], axis=0)
    )
    _map_pos: np.ndarray = np.array([_rows, _cols]).transpose([1, 2, 0])

    _map: np.ndarray = make_seed(_map_shape, CHANNEL_N)

    model: CAModel = CAModel(CHANNEL_N, CELL_FIRE_RATE, device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    output: torch.Tensor = model(
        torch.from_numpy(
            _map.reshape([1, _map_shape[0], _map_shape[1], CHANNEL_N]).astype(
                np.float32
            )
        ),
        1,
    )

    disp: displayer = displayer(_map_shape, pix_size)

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
                ).reshape([_map_shape[0], _map_shape[1], 1])
                output = torch.from_numpy(output.detach().numpy() * should_keep)
            except AttributeError:
                pass

        output = model(output, 1)

        _map = to_rgb(output.detach().numpy()[0])
        disp.update(_map)


if __name__ == "__main__":
    main()
