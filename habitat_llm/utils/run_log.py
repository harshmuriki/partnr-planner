"""Terminal mirroring that also supports a launcher-owned stdout/stderr log."""

import contextlib
import os
import sys
import time


class TeeStream:
    def __init__(self, real_stream, mirror_file):
        self._real = real_stream
        self._mirror = mirror_file

    def write(self, data):
        self._real.write(data)
        self._mirror.write(data)
        return len(data)

    def flush(self):
        self._real.flush()
        self._mirror.flush()


def _targets_file(stream, path):
    try:
        return os.path.samefile(stream.fileno(), path)
    except (AttributeError, OSError, ValueError):
        return False


@contextlib.contextmanager
def tee_to_run_log(log_path, header):
    """Mirror direct CLI output, or use the launcher's existing file descriptors.

    Opening a second writer when stdout already targets log_path would duplicate
    output and can overwrite appended bytes via independent file offsets.
    """
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    start = f"\n{'=' * 80}\n{header}\n{time.strftime('%Y-%m-%d %H:%M:%S')}\n{'=' * 80}\n"
    if _targets_file(sys.stdout, log_path) and _targets_file(sys.stderr, log_path):
        print(start, end="", flush=True)
        try:
            yield
        finally:
            print(f"\n--- end: {header} @ {time.strftime('%Y-%m-%d %H:%M:%S')} ---", flush=True)
        return
    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(start)
        logf.flush()
        try:
            with contextlib.redirect_stdout(TeeStream(sys.stdout, logf)), \
                    contextlib.redirect_stderr(TeeStream(sys.stderr, logf)):
                yield
        finally:
            logf.write(f"\n--- end: {header} @ {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            logf.flush()
