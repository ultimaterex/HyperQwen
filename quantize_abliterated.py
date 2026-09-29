"""Reproduce dbirks/Qwen3.8-27B-W4A16-AutoRound's exact recipe against
huihui-ai/Huihui-Qwen3.8-27B-abliterated instead of the official base model.

Recipe transcribed from https://huggingface.co/dbirks/Qwen3.8-27B-W4A16-AutoRound
(Reproducibility section). layer_config reconstructed from the actual tensor
names in the checkpoint (model.safetensors.index.json) since the model card
only names the BF16_FOR_IN_PROJ_AB_VISUAL_MTP variable without showing it:
  - model.language_model.layers.N.linear_attn.in_proj_a  -> BF16
  - model.language_model.layers.N.linear_attn.in_proj_b  -> BF16
  - model.visual.*                                       -> BF16 (vision tower)
  - mtp.*                                                 -> BF16 (draft head)
  - lm_head is excluded from quantization by AutoRound's own default
    (quant_lm_head=False), so it does not need an explicit entry.
AutoRound's layer_config matching is substring/regex search against each
supported layer's dotted module name (auto_round.utils.to_standard_regex),
so these short substrings are sufficient -- no need to enumerate all 48+16
layer indices.

Usage:
  quant-venv/bin/python quantize_abliterated.py --smoke-test   # iters=1, nsamples=4, sanity + peak-VRAM check
  quant-venv/bin/python quantize_abliterated.py                # full recipe (iters=200, nsamples=128)
"""

import argparse
import os

from auto_round import AutoRound

parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", default="models/Huihui-Qwen3.8-27B-abliterated")
parser.add_argument("--output-dir", default="models/Qwen3.8-27B-W4A16-AutoRound-abliterated")
parser.add_argument("--device", default="1", help="CUDA device index, or 'auto' to let accelerate offload overflow to CPU RAM")
parser.add_argument("--smoke-test", action="store_true", help="iters=1, nsamples=4 -- sanity/peak-VRAM check, not a real quant")
parser.add_argument("--iters", type=int, default=None, help="override iters (e.g. for a timing estimate run)")
parser.add_argument("--nsamples", type=int, default=None, help="override nsamples")
parser.add_argument("--low-gpu-mem-usage", action="store_true", help="offload cached block inputs to host RAM (needed to fit 24GB)")
parser.add_argument("--batch-size", type=int, default=4, help="dbirks' recipe uses 4; lower it if 24GB still OOMs mid-block-tuning")
parser.add_argument("--resume-dir", default=None, help="AR_RESUME_DIR: checkpoints per-block progress so a crash/OOM doesn't restart from block 0. Must be pre-created and writable (models/ is root-owned).")
args = parser.parse_args()

layer_config = {
    "linear_attn.in_proj_a": {"bits": 16},
    "linear_attn.in_proj_b": {"bits": 16},
    "visual.": {"bits": 16},
    "mtp.": {"bits": 16},
}

iters = 1 if args.smoke_test else 200
nsamples = 4 if args.smoke_test else 128
if args.iters is not None:
    iters = args.iters
if args.nsamples is not None:
    nsamples = args.nsamples

if args.resume_dir:
    os.environ["AR_RESUME_DIR"] = args.resume_dir

print(f"[quantize] model_dir={args.model_dir} output_dir={args.output_dir} device={args.device} "
      f"smoke_test={args.smoke_test} iters={iters} nsamples={nsamples} resume_dir={args.resume_dir}")

ar = AutoRound(
    args.model_dir,
    scheme="W4A16",
    dataset="NeelNanda/pile-10k",
    nsamples=nsamples,
    seqlen=2048,
    batch_size=args.batch_size,
    iters=iters,
    device_map=args.device if args.device == "auto" else int(args.device),
    trust_remote_code=True,
    quant_nontext_module=False,
    seed=42,
    layer_config=layer_config,
    low_gpu_mem_usage=args.low_gpu_mem_usage,
)

out_dir = args.output_dir + ("-smoketest" if args.smoke_test else "")
ar.quantize_and_save(output_dir=out_dir, format="llm_compressor")
print(f"[quantize] done -> {out_dir}")
