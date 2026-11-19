"""Small streaming H.264 videos with fragments that survive interrupted runs."""

import cv2
import imageio.v2 as imageio
import numpy as np


def resize_video_frame(frame, width=1280, height=720):
    """Fit the camera image without cropping or changing its aspect ratio."""
    source_height, source_width = frame.shape[:2]
    if (source_width, source_height) == (width, height):
        return np.ascontiguousarray(frame)
    scale = min(width / source_width, height / source_height)
    size = (max(1, round(source_width * scale)), max(1, round(source_height * scale)))
    resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x, y = (width - size[0]) // 2, (height - size[1]) // 2
    canvas[y:y + size[1], x:x + size[0]] = resized
    return canvas


def open_video_writer(path, fps=30, crf=28, preset="ultrafast"):
    """Encode incrementally; completed MP4 fragments remain playable on timeout."""
    return imageio.get_writer(
        str(path), format="FFMPEG", fps=fps, codec="libx264", quality=None,
        pixelformat="yuv420p", macro_block_size=2,
        ffmpeg_params=[
            "-preset", preset, "-crf", str(crf), "-threads", "1",
            "-g", str(max(1, round(fps / 2))),
            "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
            "-flush_packets", "1",
        ],
    )
