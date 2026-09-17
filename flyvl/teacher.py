"""InternVL3-1B (HF-native `OpenGVLab/InternVL3-1B-hf`) as teacher and as frozen LLM judge.

teacher tokens: image -> bicubic resize 448 -> ImageNet normalize -> InternViT-300M -> pixel shuffle -> MLP projector
                -> (256, 896) visual tokens = exactly what the Qwen2.5-0.5B LLM receives (16 x 16 grid).
zero-shot scorer: any (256, 896) token set is placed at the <IMG_CONTEXT> positions of a fixed prompt, and each
                CIFAR class name is scored by its summed log-probability as the assistant answer.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

from .data import CLASSES

MODEL_ID = "OpenGVLab/InternVL3-1B-hf"
QUESTION = "What is the main object in this image? Answer with one word."
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
GRID = 16


def load(device="cuda"):
    os.environ.setdefault("HF_HOME", r"D:\flyvl_data\hf")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoProcessor, InternVLForConditionalGeneration

    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = InternVLForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device).eval()
    return proc, model


@torch.no_grad()
def teacher_tokens(model, images_uint8: np.ndarray, batch: int = 32) -> torch.Tensor:
    """(N, 32, 32, 3) uint8 -> (N, 256, 896) float16 on CPU."""
    out = []
    cfg = model.config
    for s in range(0, len(images_uint8), batch):
        x = torch.from_numpy(images_uint8[s:s + batch]).permute(0, 3, 1, 2).float().div(255).cuda()
        x = F.interpolate(x, size=448, mode="bicubic", align_corners=False).clamp(0, 1)
        x = ((x - MEAN.cuda()) / STD.cuda()).to(torch.bfloat16)
        f = model.model.get_image_features(pixel_values=x, vision_feature_layer=cfg.vision_feature_layer,
                                           vision_feature_select_strategy=cfg.vision_feature_select_strategy)
        out.append(f.pooler_output.float().cpu().half())
    return torch.cat(out)


class ZeroShot:
    """Scores CIFAR class names given visual tokens."""

    def __init__(self, proc, model, classes=CLASSES):
        self.model = model
        tok = proc.tokenizer
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]}]
        prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
        prompt = prompt.replace(proc.image_token, proc.image_token * 256)
        self.prefix = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]
        self.img_id = tok.convert_tokens_to_ids(proc.image_token)
        assert int((self.prefix == self.img_id).sum()) == 256
        self.cands = [tok(c, add_special_tokens=False).input_ids for c in classes]
        L = max(len(c) for c in self.cands)
        self.cand_ids = torch.full((len(classes), L), tok.pad_token_id or 0)
        self.cand_mask = torch.zeros(len(classes), L)
        for i, c in enumerate(self.cands):
            self.cand_ids[i, :len(c)] = torch.tensor(c)
            self.cand_mask[i, :len(c)] = 1

    @torch.no_grad()
    def logprobs(self, tokens: torch.Tensor, batch: int = 8) -> torch.Tensor:
        """tokens (N, 256, 896) -> (N, n_classes) summed answer log-probs."""
        m = self.model
        emb = m.get_input_embeddings()
        P, (C, L) = len(self.prefix), self.cand_ids.shape
        ids = torch.cat([self.prefix.expand(C, -1), self.cand_ids], 1).cuda()          # (C, P+L)
        base = emb(ids)                                                                   # (C, P+L, D)
        img_pos = (self.prefix == self.img_id).nonzero().squeeze(1).cuda()
        mask = self.cand_mask.cuda()
        out = []
        for s in range(0, len(tokens), batch):
            t = tokens[s:s + batch].cuda().to(base.dtype)
            B = len(t)
            e = base.unsqueeze(0).repeat(B, 1, 1, 1)                                     # (B, C, P+L, D)
            e[:, :, img_pos] = t[:, None]
            e = e.reshape(B * C, P + L, -1)
            logits = m(inputs_embeds=e, logits_to_keep=L + 1).logits.float()             # (B*C, L+1, V)
            lp = logits[:, :L].log_softmax(-1)
            tgt = ids[:, P:].repeat(B, 1)
            ll = lp.gather(-1, tgt[..., None]).squeeze(-1) * mask.repeat(B, 1)
            out.append(ll.sum(1).reshape(B, C).cpu())
        return torch.cat(out)


def pool_tokens(tokens: torch.Tensor, g: int) -> torch.Tensor:
    """(N, 256, D) on the 16x16 grid -> (N, g*g, D) by average pooling."""
    N, _, D = tokens.shape
    x = tokens.float().reshape(N, GRID, GRID, D).permute(0, 3, 1, 2)
    return F.adaptive_avg_pool2d(x, g).permute(0, 2, 3, 1).reshape(N, g * g, D)


def unpool_tokens(tokens: torch.Tensor, g: int) -> torch.Tensor:
    """(N, g*g, D) -> (N, 256, D) by nearest upsampling to the 16x16 grid (for feeding the LLM)."""
    N, _, D = tokens.shape
    x = tokens.float().reshape(N, g, g, D).permute(0, 3, 1, 2)
    return F.interpolate(x, size=GRID, mode="nearest").permute(0, 2, 3, 1).reshape(N, GRID * GRID, D)
