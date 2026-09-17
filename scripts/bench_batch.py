"""Microbenchmark: ms per step for B in {1, 8, 16, 32, 64}; writes the chosen batch to configs/bench.json."""
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flyvl import connectome  # noqa: E402
from flyvl.sim import Sim, SimConfig  # noqa: E402

STEPS = 100
c = connectome.load()
sim = Sim(c, SimConfig())
print("W on GPU, allocated MB:", torch.cuda.memory_allocated() // 2**20)

results = {}
for B in (1, 8, 16, 32, 64):
    st = sim.zero_state(B)
    eye = torch.rand(len(sim.driven), B, device="cuda") - 0.5
    for _ in range(5):
        st = sim.step(st, eye)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(STEPS):
        st = sim.step(st, eye)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / STEPS * 1e3
    results[B] = {"ms_per_step": round(ms, 3), "ms_per_image_step": round(ms / B, 4),
                  "peak_MB": torch.cuda.max_memory_allocated() // 2**20}
    print(B, results[B], flush=True)

best = min(results, key=lambda b: results[b]["ms_per_image_step"])
(ROOT / "configs").mkdir(exist_ok=True)
(ROOT / "configs" / "bench.json").write_text(json.dumps({"results": results, "batch": best}, indent=1))
print("chosen batch:", best)
