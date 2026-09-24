"""PPO on flybody vision-guided flight. The ARM changes only the visual encoder; everything else
(PPO settings, proprio MLP, policy head, seeds, step budget) is identical across arms (program.md).

arms: blind (eyes zeroed - does vision matter at all?), cnn, and later flyvis_real / flyvis_shuffled.
usage: python flight/train.py ARM [steps] [seed] [task]
writes D:/flyvl_data/runs/flight/{task}_{arm}_s{seed}.json with the validation return.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env import FlightEnv  # noqa: E402

N_ENVS, VAL_SEEDS = 16, range(1000, 1020)
OUT = Path("D:/flyvl_data/runs/flight")


class Encoder(BaseFeaturesExtractor):
    def __init__(self, space, arm: str):
        n_prop = space["proprio"].shape[0]
        super().__init__(space, features_dim=128 + 128)
        self.arm = arm
        self.prop = torch.nn.Sequential(torch.nn.Linear(n_prop, 128), torch.nn.Tanh())
        if arm == "cnn":
            self.vis = torch.nn.Sequential(
                torch.nn.Conv2d(2, 16, 5, stride=2), torch.nn.ReLU(),   # 32 -> 14
                torch.nn.Conv2d(16, 32, 3, stride=2), torch.nn.ReLU(),  # 14 -> 6
                torch.nn.Flatten(), torch.nn.Linear(32 * 6 * 6, 128), torch.nn.ReLU())
        elif arm == "blind":
            self.vis = None
        else:
            raise ValueError(arm)

    def forward(self, obs):
        p = self.prop(obs["proprio"])
        if self.vis is None:
            return torch.cat([p, torch.zeros(len(p), 128, device=p.device)], 1)
        return torch.cat([p, self.vis(obs["eyes"].float() / 255.0)], 1)


def make(task, seed):
    return lambda: FlightEnv(task, seed)


def evaluate(model, venv_norm, task):
    """Deterministic return on held-out validation seeds, obs normalised with the TRAINING stats."""
    rets = []
    for s in VAL_SEEDS:
        env, R = FlightEnv(task, s), 0.0
        obs, _ = env.reset()
        done = False
        while not done:
            o = venv_norm.normalize_obs({k: v[None] for k, v in obs.items()})
            a, _ = model.predict(o, deterministic=True)
            obs, r, term, trunc, _ = env.step(a[0])
            R += r
            done = term or trunc
        rets.append(R)
    return float(np.mean(rets)), float(np.std(rets))


if __name__ == "__main__":
    arm = sys.argv[1]
    steps = int(float(sys.argv[2])) if len(sys.argv) > 2 else 1_000_000
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    task = sys.argv[4] if len(sys.argv) > 4 else "bumps"
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{task}_{arm}_s{seed}"
    venv = SubprocVecEnv([make(task, seed * 100 + i) for i in range(N_ENVS)])
    venv = VecNormalize(venv, norm_obs_keys=["proprio"], norm_reward=True)
    model = PPO("MultiInputPolicy", venv, n_steps=512, batch_size=1024, n_epochs=5, learning_rate=3e-4,
                gamma=0.995, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, seed=seed, device="cuda",
                policy_kwargs={"features_extractor_class": Encoder, "features_extractor_kwargs": {"arm": arm},
                               "net_arch": [256, 256]}, verbose=0)
    t0 = time.time()
    model.learn(steps)
    venv.training = False
    val, sd = evaluate(model, venv, task)
    res = {"task": task, "arm": arm, "seed": seed, "steps": steps, "val_return": val, "val_sd": sd,
           "minutes": (time.time() - t0) / 60}
    (OUT / f"{tag}.json").write_text(json.dumps(res, indent=1))
    model.save(OUT / f"{tag}.zip")
    venv.save(str(OUT / f"{tag}_vecnorm.pkl"))
    print(f"{tag} val_return {val:.3f} +- {sd:.3f} ({res['minutes']:.1f} min)", flush=True)
