"""AutoRound W4A16 quant of ajgazin/Swift-1.5-Qwen3.8-27B-Uncensored-MTP, v2 of quantize_abliterated.py's recipe.

Same layer_config (in_proj_a/b, visual.*, mtp.* kept BF16; lm_head left to prepare/'s GPTQ pass) and the same
W4A16 g128 sym scheme / llm_compressor format, with the calibration changed (2026-09-25):
  - data: 128 x 4096-token samples from prepare/build_calib_swift15.py (96 packed chat-templated generations
    from this model family -- chat, code, math, mostly with thinking traces -- plus 32 pile-10k), instead of
    128 x 2048 pile-10k only. This matches the deployment distribution and doubles the context each block
    tunes on (the long-context suite found the only depth-growing cost in the recurrent layers).
  - iters 300 instead of 200 (the previous run early-stopped blocks between iter 67 and 197).

usage: quant-venv/bin/python quantize_swift15.py [--smoke] [--device 0] [--resume-dir DIR]
"""
import argparse, os
import torch
from auto_round import AutoRound

p = argparse.ArgumentParser()
p.add_argument("--model-dir", default="/mnt/nvme1/models_archive/sources/Swift-1.5-Qwen3.8-27B-Uncensored-MTP")
p.add_argument("--calib", default="/mnt/nvme1/models_archive/sources/calib_swift15_128x4096.pt")
p.add_argument("--output-dir", default="models/Swift-1.5-Qwen3.8-27B-Uncensored-W4A16-AutoRound")
p.add_argument("--device", default="0")
p.add_argument("--iters", type=int, default=300)
p.add_argument("--batch-size", type=int, default=2)
p.add_argument("--grad-accum", type=int, default=2, help="effective batch = batch-size x grad-accum (4, as the original recipe)")
p.add_argument("--smoke", action="store_true", help="iters=2 on 8 samples: VRAM/plumbing check only")
p.add_argument("--memcheck", action="store_true", help="all samples, iters=2: peak-VRAM check of the real sample count")
p.add_argument("--resume-dir", default=None)
a = p.parse_args()

layer_config = {
    "linear_attn.in_proj_a": {"bits": 16},
    "linear_attn.in_proj_b": {"bits": 16},
    "visual.": {"bits": 16},
    "mtp.": {"bits": 16},
}
calib = torch.load(a.calib)
iters = a.iters
if a.smoke:
    calib, iters = calib[:8], 2
if a.memcheck:
    iters = 2
if a.resume_dir:
    os.makedirs(a.resume_dir, exist_ok=True)
    os.environ["AR_RESUME_DIR"] = a.resume_dir
print(f"[quantize] {a.model_dir} -> {a.output_dir} samples={len(calib)}x{calib[0].shape[-1]} iters={iters} "
      f"bs={a.batch_size}x{a.grad_accum} device={a.device} smoke={a.smoke}", flush=True)

ar = AutoRound(
    a.model_dir,
    scheme="W4A16",
    dataset=calib,
    nsamples=len(calib),
    seqlen=calib[0].shape[-1],
    batch_size=a.batch_size,
    gradient_accumulate_steps=a.grad_accum,
    iters=iters,
    device_map=int(a.device),
    trust_remote_code=True,
    quant_nontext_module=False,
    seed=42,
    layer_config=layer_config,
    low_gpu_mem_usage=True,
)
out = a.output_dir + ("-smoke" if a.smoke else "-memcheck" if a.memcheck else "")
ar.quantize_and_save(output_dir=out, format="llm_compressor")
print(f"[quantize] done -> {out}", flush=True)
