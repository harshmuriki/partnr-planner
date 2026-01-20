#!/usr/bin/env python3
"""Object visualization component."""

from typing import Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from .utils import wrap_text


class Object:
    """Represents an object in the visualization."""
    
    def __init__(self, object_id: str, color: str, config: dict):
        self.object_id = object_id
        self.color = color
        self.config = config
        self.center_position: Optional[Tuple[float, float]] = None
    
    @property
    def width(self) -> float:
        return self.config.get("width", 100)
    
    @property
    def height(self) -> float:
        return self.config.get("height", 100)
    
    def plot(self, ax: plt.Axes, origin: Tuple[float, float]) -> plt.Axes:
        """Plot the object at the given origin."""
        # Create rectangle for object
        rect = FancyBboxPatch(
            (origin[0], origin[1]),
            self.width,
            self.height,
            edgecolor="white",
            facecolor=self.color,
            linewidth=0,
            boxstyle=f"Round, pad=0, rounding_size={self.config.get('rounding_size', 10)}",
            alpha=1.0,
        )
        ax.add_patch(rect)
        
        # Update center position
        self.center_position = (
            origin[0] + self.width / 2,
            origin[1] + self.height / 2,
        )
        
        # Add text label
        text_position = (
            self.center_position[0],
            self.center_position[1] + self.config.get("text_margin", -150),
        )
        
        wrapped_text = wrap_text(self.object_id, self.config.get("max_chars_per_line", 8))
        ax.annotate(
            wrapped_text,
            xy=text_position,
            ha="center",
            va="center",
            fontsize=self.config.get("text_size", 14),
            color="white",
            zorder=float("inf"),
        )
        
        return ax

