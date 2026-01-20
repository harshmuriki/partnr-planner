#!/usr/bin/env python3
"""Room visualization component."""

from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
from .constants import ROOM_COLOR
from .object import Object
from .receptacle import Receptacle
from .utils import wrap_text


class Room:
    """Represents a room in the visualization."""
    
    def __init__(
        self,
        room_id: str,
        receptacles: List[Receptacle],
        objects: List[Object],
        object_to_recep: dict,
        config: dict,
    ):
        self.room_id = room_id
        self.receptacles = receptacles
        self.objects = objects
        self.object_to_recep = object_to_recep
        self.config = config
        self.center_position: Optional[Tuple[float, float]] = None
        
        self.init_size()
    
    def init_size(self) -> None:
        """Calculate room dimensions based on contents."""
        # Width based on receptacles
        min_width = self.config.get("min_width", 450)
        if self.receptacles:
            receptacle_width = sum(r.width for r in self.receptacles)
            min_width = max(min_width, receptacle_width)
        
        # Add padding
        self.room_width = min_width + self.config.get("left_pad", 20) + self.config.get("right_pad", 20)
        self.width = self.room_width + 2 * self.config.get("horizontal_margin", 10)
        
        # Height
        self.room_height = self.config.get("min_height", 700)
        if self.objects:
            self.room_height = int(self.room_height * 1.5)  # More space for objects
        
        self.room_height += self.config.get("top_pad", 100) + self.config.get("bottom_pad", 175)
        self.height = self.room_height + 2 * self.config.get("vertical_margin", 10)
    
    def find_receptacle_by_id(self, receptacle_id: str) -> Optional[Receptacle]:
        """Find a receptacle by its ID."""
        for recep in self.receptacles:
            if recep.receptacle_id == receptacle_id:
                return recep
        return None
    
    def plot(
        self,
        ax: plt.Axes,
        origin: Tuple[float, float],
        target_width: Optional[float] = None,
    ) -> plt.Axes:
        """Plot the room at the given origin."""
        actual_origin = (
            origin[0] + self.config.get("horizontal_margin", 10),
            origin[1] + self.config.get("vertical_margin", 10),
        )
        
        # Adjust width if target specified
        if target_width:
            extra_pad = max(0, (target_width - self.room_width - 2 * self.config.get("horizontal_margin", 10)) / 2)
            self.room_width += 2 * extra_pad
            self.width = self.room_width + 2 * self.config.get("horizontal_margin", 10)
        
        # Draw room rectangle
        rect = plt.Rectangle(
            actual_origin,
            self.room_width,
            self.room_height,
            edgecolor="white",
            linewidth=1,
            facecolor=ROOM_COLOR,
            alpha=self.config.get("box_alpha", 1.0),
            zorder=-1,
        )
        ax.add_patch(rect)
        
        # Room label at bottom
        text_x = actual_origin[0] + self.room_width / 2
        text_y = actual_origin[1] + self.config.get("bottom_pad", 175) / 4
        wrapped_text = wrap_text(self.room_id, self.config.get("max_chars_per_line", 13))
        ax.text(
            text_x,
            text_y,
            wrapped_text,
            ha="center",
            va="bottom",
            fontsize=self.config.get("text_size", 14),
            color="white",
            zorder=float("inf"),
        )
        
        # Plot receptacles
        if self.receptacles:
            receptacle_width = sum(r.width for r in self.receptacles)
            spacing = (self.room_width - receptacle_width) / (len(self.receptacles) + 1)
            offset = actual_origin[0] + spacing
            
            for receptacle in self.receptacles:
                receptacle.plot(
                    ax,
                    origin=(offset, actual_origin[1] + self.config.get("bottom_pad", 175)),
                )
                offset += receptacle.width + spacing
        
        # Plot objects
        if self.objects:
            # Floor objects (not on receptacles)
            floor_objects = [
                obj for obj in self.objects
                if obj.object_id not in self.object_to_recep or self.object_to_recep[obj.object_id] == "floor"
            ]
            
            if floor_objects:
                total_obj_width = sum(obj.width for obj in floor_objects)
                obj_spacing = (self.room_width - total_obj_width) / (len(floor_objects) + 1)
                obj_offset = actual_origin[0] + obj_spacing
                
                for obj in floor_objects:
                    obj.plot(
                        ax,
                        origin=(
                            obj_offset,
                            actual_origin[1] + self.room_height * 0.7,
                        ),
                    )
                    obj_offset += obj.width + obj_spacing
            
            # Objects on receptacles
            for obj in self.objects:
                if obj.object_id in self.object_to_recep:
                    recep_id = self.object_to_recep[obj.object_id]
                    if recep_id != "floor":
                        receptacle = self.find_receptacle_by_id(recep_id)
                        if receptacle and receptacle.next_object_position:
                            obj.plot(ax, receptacle.next_object_position)
                            # Update position for next object (stack vertically)
                            receptacle.next_object_position = (
                                receptacle.next_object_position[0],
                                receptacle.next_object_position[1] + obj.height + 50,
                            )
        
        # Set center position for room
        self.center_position = (
            actual_origin[0] + self.room_width / 2,
            actual_origin[1] + self.room_height / 2,
        )
        
        return ax

