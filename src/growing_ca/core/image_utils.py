"""Utilities for processing image sprite sheets."""

import sys
from pathlib import Path

import imageio.v3 as imageio
import numpy as np


def load_image_from_sheet(
    sheet_path: str, index: int, sprite_width: int = 40
) -> np.ndarray:
    """Load a single image from a sprite sheet.

    Args:
        sheet_path: Path to the image sprite sheet
        index: Index of the image in the sprite sheet (0-based)
        sprite_width: Width of each sprite in pixels (default: 40)

    Returns:
        Image as a float32 numpy array with values in [0, 1]
    """
    im = imageio.imread(sheet_path)
    image = np.array(
        im[:, index * sprite_width : (index + 1) * sprite_width].astype(np.float32)
    )
    image /= 255.0
    return image


def split_image_sheet(
    input_path: str,
    output_dir: str,
    sprite_width: int = 40,
    output_prefix: str = "image",
) -> list[str]:
    """Split an image sprite sheet into individual images.

    Args:
        input_path: Path to the input sprite sheet
        output_dir: Directory to save individual images
        sprite_width: Width of each sprite in pixels (default: 40)
        output_prefix: Prefix for output filenames (default: "image")

    Returns:
        List of paths to the created image files
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load the sprite sheet
    im = imageio.imread(input_path)

    # Calculate number of images
    height, width = im.shape[:2]
    num_images = width // sprite_width

    output_files = []

    # Split and save each image
    for i in range(num_images):
        image = im[:, i * sprite_width : (i + 1) * sprite_width]
        output_file = output_path / f"{output_prefix}_{i}.png"
        imageio.imwrite(output_file, image)
        output_files.append(str(output_file))
        print(f"Saved {output_file}")

    return output_files


def main() -> None:
    """CLI entry point for splitting image sheets."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Split image sprite sheet into individual images"
    )
    parser.add_argument(
        "input_path", type=str, help="Path to the input image sprite sheet"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/images",
        help="Output directory for individual images (default: data/images)",
    )
    parser.add_argument(
        "--sprite-width",
        type=int,
        default=40,
        help="Width of each sprite in pixels (default: 40)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="image",
        help="Prefix for output filenames (default: image)",
    )

    args = parser.parse_args()

    # Check if input file exists
    if not Path(args.input_path).exists():
        print(f"Error: Input file not found at {args.input_path}")
        sys.exit(1)

    print(f"Splitting image sheet: {args.input_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sprite width: {args.sprite_width}")
    print("-" * 60)

    output_files = split_image_sheet(
        args.input_path,
        args.output_dir,
        args.sprite_width,
        args.output_prefix,
    )

    print("-" * 60)
    print(f"Successfully split {len(output_files)} images")


if __name__ == "__main__":
    main()
