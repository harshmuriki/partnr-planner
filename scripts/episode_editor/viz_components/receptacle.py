#!/usr/bin/env python3
"""Receptacle visualization component."""

from typing import Optional, Tuple
import os
import matplotlib.pyplot as plt
from PIL import Image
from .constants import receptacle_color_map, RECEPTACLE_ICONS_PATH
from .utils import resize_icon_height, add_tint_to_rgb


class Receptacle:
    """Represents a receptacle/furniture in the visualization."""
    
    def __init__(self, receptacle_id: str, config: dict, project_root: str):
        self.receptacle_id = receptacle_id
        self.config = config
        self.project_root = project_root
        self.center_position: Optional[Tuple[float, float]] = None
        self.top_position: Optional[Tuple[float, float]] = None
        self.next_object_position: Optional[Tuple[float, float]] = None
        
        # Determine icon path
        recep_type = "_".join(receptacle_id.split("_")[:-1])
        icon_filename = f"{recep_type}@2x.png"
        self.icon_path = os.path.join(project_root, RECEPTACLE_ICONS_PATH, icon_filename)
        
        # Fallback to default icon if specific one doesn't exist
        if not os.path.exists(self.icon_path):
            self.icon_path = os.path.join(project_root, RECEPTACLE_ICONS_PATH, "chair@2x.png")
        
        self.init_size()
    
    @property
    def horizontal_margin(self) -> float:
        return self.config.get("horizontal_margin", 5)
    
    def init_size(self) -> None:
        """Initialize receptacle size based on icon."""
        if os.path.exists(self.icon_path):
            icon = self.get_icon(add_tint=False)
            icon_width, icon_height = icon.size
            self.width = icon_width + 2 * self.horizontal_margin
            self.height = icon_height
        else:
            # Default size if icon doesn't exist
            self.width = 200
            self.height = 300
    
    def get_icon(self, add_tint: bool = True) -> Image.Image:
        """Load and optionally tint the receptacle icon."""
        if not os.path.exists(self.icon_path):
            # Return a placeholder image
            return Image.new("RGBA", (200, 300), (100, 100, 100, 255))
        
        icon = Image.open(self.icon_path)
        icon = resize_icon_height(icon, self.config.get("target_height", 500))
        
        if add_tint:
            recep_type = "_".join(self.receptacle_id.split("_")[:-1])
            color = receptacle_color_map.get(recep_type, (0.7, 0.7, 0.7))
            tint_color = tuple(int(255 * i) for i in color)
            icon = add_tint_to_rgb(icon, tint_color=tint_color)
        
        return icon
    
    def plot(self, ax: plt.Axes, origin: Tuple[float, float]) -> plt.Axes:
        """Plot the receptacle at the given origin."""
        icon = self.get_icon()
        receptacle_width, receptacle_height = icon.size
        
        # Display icon
        ax.imshow(
            icon,
            extent=(
                origin[0] + self.horizontal_margin,
                origin[0] + receptacle_width + self.horizontal_margin,
                origin[1],
                origin[1] + receptacle_height,
            ),
        )
        
        # Set positions for objects to be placed on/in receptacle
        self.center_position = (
            origin[0] + self.width / 2,
            origin[1] + receptacle_height / 2,
        )
        
        self.top_position = (
            origin[0] + self.width / 2,
            origin[1] + receptacle_height + self.config.get("object_margin", 60),
        )
        
        # Position for next object to be stacked
        self.next_object_position = self.top_position
        
        # Add label
        label_x = origin[0] + self.width / 2
        label_y = origin[1] - 10
        ax.text(
            label_x,
            label_y,
            self.receptacle_id,
            ha="center",
            va="top",
            fontsize=10,
            color="white",
            alpha=0.8,
        )
        
        return ax

