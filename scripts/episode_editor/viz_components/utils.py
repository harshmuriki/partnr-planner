#!/usr/bin/env python3
"""Utility functions for visualization."""

import re
from typing import Tuple
from PIL import Image, ImageChops


def wrap_text(text: str, max_chars_per_line: int) -> str:
    """Wrap text to fit within max_chars_per_line."""
    # Remove trailing numbers after underscore
    text = re.sub(r"_(\d+)", "", text)
    # Replace underscores and slashes with spaces
    text = text.replace("/", "_").replace(" ", "_")
    names = text.split("_")
    
    current_line = ""
    wrapped_text = []
    for name in names:
        name = name.strip()
        if len(current_line + name) <= max_chars_per_line:
            current_line += name + " "
        else:
            wrapped_text.append(current_line.strip())
            current_line = name + " "
    wrapped_text.append(current_line.strip())
    return "\n".join(wrapped_text).strip()


def resize_icon_height(icon: Image.Image, target_height: float) -> Image.Image:
    """Resize icon to target height while maintaining aspect ratio."""
    width, height = icon.size
    scaling_factor = target_height / height
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    return icon.resize((new_width, new_height))


def add_tint_to_rgb(image: Image.Image, tint_color: Tuple) -> Image.Image:
    """Add a tint color to an image."""
    r, g, b, alpha = image.split()
    tint = Image.new("RGB", image.size, tint_color)
    tinted_rgb = ImageChops.screen(tint.convert("RGB"), image.convert("RGB"))
    
    return Image.merge(
        "RGBA",
        (
            tinted_rgb.split()[0],
            tinted_rgb.split()[1],
            tinted_rgb.split()[2],
            alpha,
        ),
    )

