import numpy as np


class displayer:
    def __init__(
        self,
        _map_shape: tuple[int, int],
        pix_size: int,
        has_gap: bool = False,
        target_image: np.ndarray | None = None,
    ) -> None:
        """
        _map_size: tuple
        color_map: a list indicates the color to each index.
                   0 : empty block, should always white
                   1+: varies building types
        target_image: optional target image to display side by side
        """
        import pygame

        pygame.init()
        clock: pygame.time.Clock = pygame.time.Clock()
        clock.tick(60)

        self.has_gap: bool = has_gap
        self.pix_size: int = pix_size
        self.target_image: np.ndarray | None = target_image

        # Calculate window size based on whether we have a target image
        if target_image is not None:
            padding = 10
            # Use actual target image dimensions instead of hardcoded size
            image_display_width = target_image.shape[1] * pix_size
            image_display_height = target_image.shape[0] * pix_size
            total_width = _map_shape[1] * self.pix_size + padding + image_display_width
            total_height = max(_map_shape[0] * self.pix_size, image_display_height)
        else:
            total_width = _map_shape[1] * self.pix_size
            total_height = _map_shape[0] * self.pix_size

        self.screen: pygame.Surface = pygame.display.set_mode(
            (total_width, total_height)
        )
        self._map_shape = _map_shape

    def update(self, _map: np.ndarray) -> None:
        import pygame

        self.screen.fill((255, 255, 255))

        # Draw the CA map on the left side
        for i in range(_map.shape[0]):
            for j in range(_map.shape[1]):
                x: int = j * self.pix_size + int(self.pix_size / 2)
                y: int = i * self.pix_size + int(self.pix_size / 2)
                if self.has_gap:
                    size: int = min(int(self.pix_size * 0.75), self.pix_size - 2)
                else:
                    size = self.pix_size
                s: pygame.Surface = pygame.Surface((size, size))
                c: np.ndarray = (_map[i, j] * 256).astype(int)[:3]
                s.fill(tuple(c))
                self.screen.blit(s, (x - int(size / 2), y - int(size / 2)))

        # Draw the target image on the right side if provided
        if self.target_image is not None:
            padding = 10
            image_x_offset = self._map_shape[1] * self.pix_size + padding

            for i in range(self.target_image.shape[0]):
                for j in range(self.target_image.shape[1]):
                    image_x: int = (
                        image_x_offset + j * self.pix_size + int(self.pix_size / 2)
                    )
                    image_y: int = i * self.pix_size + int(self.pix_size / 2)
                    image_s: pygame.Surface = pygame.Surface(
                        (self.pix_size, self.pix_size)
                    )
                    image_c: np.ndarray = np.clip(
                        self.target_image[i, j] * 255, 0, 255
                    ).astype(int)[:3]
                    image_s.fill(tuple(image_c))
                    self.screen.blit(
                        image_s,
                        (
                            image_x - int(self.pix_size / 2),
                            image_y - int(self.pix_size / 2),
                        ),
                    )

        pygame.display.update()
