#!/usr/bin/env python3
"""Scene visualization component."""

from typing import List, Tuple
import matplotlib.pyplot as plt
from .constants import BACKGROUND_COLOR
from .room import Room


class Scene:
    """Represents the entire scene containing all rooms."""
    
    def __init__(self, rooms: List[Room], config: dict):
        self.rooms = rooms
        self.config = config
        self.width = 0
        self.height = 0
    
    def plot(self) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the entire scene with all rooms."""
        fig, ax = plt.subplots(figsize=(20, 12))
        fig.patch.set_facecolor(BACKGROUND_COLOR)
        ax.set_facecolor(BACKGROUND_COLOR)
        
        target_width = self.config.get("target_width", 6000)
        
        # Layout rooms in rows
        current_row_width = 0
        current_row_height = 0
        current_row_rooms = []
        max_width = 0
        
        for room in self.rooms:
            if current_row_width + room.width <= target_width:
                current_row_rooms.append(room)
                current_row_width += room.width
            else:
                # Plot current row
                row_width, new_height = self._plot_room_row(
                    ax, current_row_rooms, current_row_height, target_width
                )
                max_width = max(max_width, row_width)
                current_row_height = new_height
                
                # Start new row
                current_row_rooms = [room]
                current_row_width = room.width
        
        # Plot last row
        if current_row_rooms:
            row_width, new_height = self._plot_room_row(
                ax, current_row_rooms, current_row_height, target_width
            )
            max_width = max(max_width, row_width)
            current_row_height = new_height
        
        # Set final dimensions
        self.width = max_width
        self.height = abs(current_row_height)
        
        ax.set_xlim(0, self.width)
        ax.set_ylim(current_row_height, 0)
        ax.axis("off")
        
        return fig, ax
    
    def _plot_room_row(
        self,
        ax: plt.Axes,
        rooms: List[Room],
        current_height: float,
        target_width: float,
    ) -> Tuple[float, float]:
        """Plot a row of rooms."""
        # Calculate heights - move down
        max_room_height = max(room.height for room in rooms)
        new_height = current_height - max_room_height
        
        # Make all rooms in row same height
        for room in rooms:
            room.height = max_room_height
            room.room_height = max_room_height - 2 * room.config.get("vertical_margin", 10)
        
        # Distribute width
        total_room_width = sum(room.width for room in rooms)
        if total_room_width < target_width:
            # Redistribute width proportionally
            scale_factor = target_width / total_room_width
            for room in rooms:
                room.width = room.width * scale_factor
                room.room_width = room.room_width * scale_factor
        
        # Plot rooms left to right
        current_x = 0
        for room in rooms:
            room.plot(ax, origin=(current_x, new_height))
            current_x += room.width
        
        return current_x, new_height

