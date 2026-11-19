"""Exercise actual video encoding and recovery without Habitat or a GPU."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

MODULE = Path(__file__).resolve().parents[1] / "habitat_llm/utils/episode_video.py"
spec = importlib.util.spec_from_file_location("episode_video", MODULE)
video = importlib.util.module_from_spec(spec)
spec.loader.exec_module(video)
FFPROBE = Path(sys.executable).parent / "ffprobe"


class EpisodeVideoTests(unittest.TestCase):
    def probe(self, path):
        result = subprocess.check_output([
            str(FFPROBE), "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,nb_read_frames,pix_fmt",
            "-of", "json", str(path),
        ], text=True)
        return json.loads(result)["streams"][0]

    def test_resize_preserves_aspect_ratio(self):
        frame = np.full((512, 512, 3), 255, dtype=np.uint8)
        resized = video.resize_video_frame(frame)
        self.assertEqual(resized.shape, (720, 1280, 3))
        self.assertTrue((resized[:, :280] == 0).all())
        self.assertTrue((resized[:, 280:1000] == 255).all())
        self.assertTrue((resized[:, 1000:] == 0).all())

    def test_encode_720p_and_single_starting_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "one.mp4"
            with video.open_video_writer(path) as writer:
                writer.append_data(np.zeros((720, 1280, 3), dtype=np.uint8))
            info = self.probe(path)
            self.assertEqual((info["width"], info["height"]), (1280, 720))
            self.assertEqual(info["codec_name"], "h264")
            self.assertEqual(info["pix_fmt"], "yuv420p")
            self.assertEqual(int(info["nb_read_frames"]), 1)

    def test_completed_fragments_survive_process_group_kill(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "interrupted.mp4"
            ready = Path(folder) / "ready"
            script = """
import importlib.util, pathlib, sys, time
import numpy as np
spec=importlib.util.spec_from_file_location('video',sys.argv[1])
video=importlib.util.module_from_spec(spec); spec.loader.exec_module(video)
writer=video.open_video_writer(sys.argv[2])
for i in range(120):
    frame=np.full((720,1280,3), i, dtype=np.uint8)
    writer.append_data(frame)
pathlib.Path(sys.argv[3]).write_text('ready')
time.sleep(60)
"""
            process = subprocess.Popen([sys.executable, "-c", script, str(MODULE), str(path), str(ready)],
                                       start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            try:
                deadline = time.monotonic() + 30
                while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.1)
                self.assertTrue(ready.exists(), "Encoder did not reach ready state")
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
                info = self.probe(path)
                self.assertEqual((info["width"], info["height"]), (1280, 720))
                self.assertGreaterEqual(int(info["nb_read_frames"]), 60)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                process.stderr.close()


if __name__ == "__main__":
    unittest.main()
