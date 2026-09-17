"""Evoked whole-CNS features for images under FlyVL-simple-v1 (see PROTOCOL.md).

feature[neuron] = mean over the 4 drift directions of (image branch - blank branch), where
  LIF units:    spike count over stim_steps
  graded units: mean x over stim_steps
Every branch starts from the same gray warm-up snapshot. The blank branch is computed once per graph.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

from . import connectome, frozen
from .sim import Sim
from .stimulus import GRAY, Retina

LUMA = torch.tensor([0.299, 0.587, 0.114])


def to_luma(images_uint8: np.ndarray) -> torch.Tensor:
    """(B, H, W, 3) uint8 -> (B, 1, H, W) float luminance in [0, 1]."""
    x = torch.from_numpy(images_uint8).float() / 255.0
    return (x @ LUMA)[:, None]


def load_graph(c: connectome.Connectome, name: str, cfg) -> sparse.csr_matrix:
    if name == "real":
        return connectome.effective_W(c, cfg)
    return sparse.load_npz(connectome.DATA_ROOT / "graphs" / f"{name}.npz").tocsr()


class Extractor:
    def __init__(self, c: connectome.Connectome, graph: str = "real", device: str = "cuda"):
        self.cfg, self.eye_cfg, self.spec = frozen.load()
        self.c, self.graph, self.device = c, graph, device
        r16 = c.types(["R1-6"])
        self.driven = r16[c.column[r16, 0] >= 0]
        self.sim = Sim(c, self.cfg, device=device, W=load_graph(c, graph, self.cfg), driven=self.driven)
        self.retina = Retina(c, self.driven, self.eye_cfg, self.cfg.dt, device=device)
        self.T = self.spec["stim_steps"]
        self.dirs = self.eye_cfg.drift_directions
        self.snapshot = self._warm()
        self.blank_s, self.blank_x, _ = self._run(None)

    def _warm(self):
        st = self.sim.zero_state(1)
        for _ in range(self.spec["warm_steps"]):
            st = self.sim.step(st)
        return st

    @torch.no_grad()
    def _run(self, images: torch.Tensor | None):
        """images (B, 1, H, W) or None for the blank branch. Returns per-(direction x image) sums."""
        B = 1 if images is None else images.shape[0] * len(self.dirs)
        st = self.snapshot.expand(B)
        self.retina.reset(B)
        sum_s = torch.zeros(self.sim.Nl, B, device=self.device)
        sum_x = torch.zeros(self.sim.Ng, B, device=self.device)
        tail_s = torch.zeros(self.sim.Nl, B, device=self.device)
        for k in range(self.T):
            t = k * self.cfg.dt
            if images is None:
                lum = torch.full((len(self.driven), 1), GRAY, device=self.device)
            else:
                lum = torch.cat([self.retina.sample_images(images, t, d) for d in self.dirs], 1)
            st = self.sim.step(st, self.retina.transduce(lum))
            sum_s += st.s
            sum_x += st.x
            if k >= self.T - 10:
                tail_s += st.s
        tail_frac = ((tail_s / (10 * self.cfg.dt)) > 40).float().mean(0)       # (B,)
        return sum_s, sum_x / self.T, tail_frac

    @torch.no_grad()
    def features(self, images: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """images (B, 1, H, W) luminance on device -> (features (B, N) float32, tail_frac (B, 4))."""
        B, D = images.shape[0], len(self.dirs)
        sum_s, mean_x, tail = self._run(images)
        ds = (sum_s - self.blank_s).reshape(self.sim.Nl, D, B).mean(1)     # directions were concatenated
        dx = (mean_x - self.blank_x).reshape(self.sim.Ng, D, B).mean(1)
        out = torch.empty(self.c.n, B, device=self.device)
        out[torch.as_tensor(self.sim.li, device=self.device)] = ds
        out[torch.as_tensor(self.sim.gi, device=self.device)] = dx
        return out.T.cpu().numpy(), tail.reshape(D, B).T.cpu().numpy()

    def spec_record(self) -> dict:
        return {"graph": self.graph, "frozen": self.spec, "sim": self.sim.spec()}


def extract_to_memmap(ex: Extractor, images_uint8: np.ndarray, out_dir: Path, batch: int = 64) -> np.ndarray:
    """Resumable chunked extraction. Writes features.fp16.npy (N_img, N_neurons), tail_frac.npy, progress.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(images_uint8)
    feat_path, tail_path, prog_path = out_dir / "features.fp16.npy", out_dir / "tail_frac.npy", out_dir / "progress.json"
    done = json.loads(prog_path.read_text())["done"] if prog_path.exists() and feat_path.exists() else 0
    mode = "r+" if done else "w+"
    feats = np.lib.format.open_memmap(feat_path, mode=mode, dtype=np.float16, shape=(n, ex.c.n))
    tails = np.lib.format.open_memmap(tail_path, mode=mode, dtype=np.float32, shape=(n, len(ex.dirs)))
    (out_dir / "spec.json").write_text(json.dumps(ex.spec_record(), indent=1, default=str))
    for start in range(done, n, batch):
        imgs = to_luma(images_uint8[start:start + batch]).to(ex.device)
        f, t = ex.features(imgs)
        feats[start:start + len(f)] = f.astype(np.float16)
        tails[start:start + len(f)] = t
        feats.flush()
        tails.flush()
        prog_path.write_text(json.dumps({"done": start + len(f), "total": n}))
    return feats
