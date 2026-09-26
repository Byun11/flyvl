"""Import flyvis on Windows.

datamate (flyvis's storage layer) opens a new .h5 file, fails to find its "data" key, then unlinks the
file while its handle is still open - fine on POSIX, WinError 32 on Windows. This replaces that writer
with one that closes the handle first. Import this module instead of flyvis directly.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("FLYVIS_ROOT_DIR", r"D:\flyvl_data\flyvis")

import datamate.io as _io  # noqa: E402
import h5py as _h5  # noqa: E402


def _write_h5(path: Path, val) -> None:
    val = np.asarray(val)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        path.rmdir()
    elif path.exists():
        path.unlink()
    with _h5.File(path, libver="latest", mode="w") as f:
        f["data"] = val


_io._write_h5 = _write_h5
import datamate.directory as _dir  # noqa: E402

if hasattr(_dir, "_write_h5"):
    _dir._write_h5 = _write_h5

import torch  # noqa: E402

import flyvis  # noqa: E402,F401

# flyvis calls torch.set_default_device("cuda") at import, which breaks every CPU-seeded generator in this
# project. Restore CPU as the default; flyvis calls are wrapped in `with torch.device("cuda")` instead.
torch.set_default_device("cpu")
