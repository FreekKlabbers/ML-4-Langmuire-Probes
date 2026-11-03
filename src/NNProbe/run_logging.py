import platform
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import torch


class _Tee:
    """Write to the original stream and the log file at the same time."""

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, text):
        self.stream.write(text)
        self.log_file.write(text)
        self.log_file.flush()  # keep the log current while the run is in progress
        return len(text)

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _git_revision() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit} (uncommitted changes)" if dirty else commit


def _format_settings(settings: dict) -> str:
    width = max(len(key) for key in settings)
    lines = []
    for key, value in settings.items():
        text = str(value)
        if "\n" in text:  # multi-line values (e.g. a model repr) go in an indented block
            text = "\n" + "\n".join("    " + line for line in text.splitlines())
        lines.append(f"{key:<{width}} : {text}")
    return "\n".join(lines)


@contextmanager
def log_run(script_name: str, settings: dict, output_dir: str | Path = "output"):
    """
    Copy everything printed to stdout/stderr inside the block to
    `output_dir/<YYYY-MM-DD_HH-MM-SS>_<script_name>.log`, preceded by a header with
    the run settings and environment. The terminal output is unchanged.
    """
    start = datetime.now()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{start:%Y-%m-%d_%H-%M-%S}_{script_name}.log"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        device += f" ({torch.cuda.get_device_name(0)})"
    environment = {
        "script": script_name,
        "started": f"{start:%Y-%m-%d %H:%M:%S}",
        "git revision": _git_revision(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
    }

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("=" * 60 + "\nRUN INFO\n" + "=" * 60 + "\n")
        log_file.write(_format_settings(environment) + "\n\n")
        log_file.write("=" * 60 + "\nSETTINGS\n" + "=" * 60 + "\n")
        log_file.write(_format_settings(settings) + "\n\n")
        log_file.write("=" * 60 + "\nOUTPUT\n" + "=" * 60 + "\n")
        log_file.flush()

        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(original_stdout, log_file)
        sys.stderr = _Tee(original_stderr, log_file)
        print(f"Logging output to {log_path}")
        status = "finished"
        try:
            yield log_path
        except BaseException:
            status = "FAILED"
            # the interpreter prints the traceback to the terminal; add it to the log
            log_file.write("\n" + traceback.format_exc())
            raise
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr
            elapsed = time.strftime(
                "%H:%M:%S", time.gmtime((datetime.now() - start).total_seconds())
            )
            log_file.write(f"\nRun {status} after {elapsed}\n")
