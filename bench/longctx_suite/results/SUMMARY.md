# Long-context quantization suite (2026-09-24), RTX 3090

Question: does our checkpoint's int4 `lm_head` (published: int8 embed, int4 lm_head, int4 linear-attention
projections) cause the long-context / multi-compaction degradation reported on the HF discussion (~8% from
lm_head quant, ~4% from quantized linear attention, mostly DNFs from reasoning loops)?

Variants (same body, same source weights, verified against the BF16 source):
- **A**: published checkpoint (`...-gptq-full-mtpfix`)
- **B**: A + bf16 `lm_head` (`...-mtpfix-bf16lmhead`). Costs 1.9 GB VRAM: `-2x` max context 196,608 → 153,920.
- **D**: B + bf16 linear-attention projections (`in_proj_qkv`, `in_proj_z`, `out_proj`, all 48 layers; +7.7 GB,
  does not fit 24 GB with a usable KV cache; tested with `--cpu-offload-gb=10`, T1 only)

Serving: T1 `SPEC=off CTX=long PREFIX_CACHE=0`; T2/T3 the production `-2x` config (MTP, fp8 KV, INT8_ACT,
thinking on, server-default sampling, fixed seeds, 2 concurrent). vLLM 0.29 image.

## T1: teacher-forced NLL by context depth (6 PG19 books + 2 code docs, 64k tokens each)

| | 0-2k | 2-8k | 8-16k | 16-32k | 32-48k | 48-64k |
|---|---|---|---|---|---|---|
| A | 2.0673 | 1.8668 | 1.8457 | 1.8107 | 1.8012 | 1.8094 |
| B | 2.0594 | 1.8583 | 1.8368 | 1.8021 | 1.7928 | 1.8010 |
| D | 2.0579 | 1.8564 | 1.8353 | 1.7997 | 1.7906 | 1.7979 |
| A−B | 0.0080 | 0.0085 | 0.0089 | 0.0086 | 0.0085 | 0.0084 |
| B−D | 0.0015 | 0.0020 | 0.0015 | 0.0024 | 0.0022 | 0.0031 |

The lm_head cost is ~0.45% and flat with depth (no compounding). The linear-attention cost is ~0.1-0.2% and grows
slightly with depth.

## T2: state tracking across compaction cycles (12 episodes; model sees only its own previous summary)

| | normal (6 cycles × 8 ops, 16k-token chunks) | hard (10 cycles × 16 ops, 24k-token chunks) |
|---|---|---|
| A | 12/12 correct, cycle acc 1.000 | 9/12 correct, 3 wrong, 0 DNF, cycle acc 0.850 |
| B | 10/12 correct, cycle acc 0.944 | 8/12 correct, 3 wrong, 1 DNF (length), cycle acc 0.786 |

Paired by seed: B failed 5 episodes that A passed; A failed 2 that B passed. At n=12 that is noise (sign test
p≈0.45), and it points the opposite way from the claim.

## T3: MATH-500 level 5 behind a ~40k-token code context (40 problems, 16k-token budget)

| | correct | wrong | DNF (hit budget) | reasoning loops |
|---|---|---|---|---|
| A | 38 | 0 | 2 (#4, #25) | 0 |
| B | 37 | 1 | 2 (#19, #25) | 0 |

Scoring was re-normalized identically for both, and every remaining mismatch was checked by hand.

## Conclusion

On one 3090, at these lengths (≤64k teacher-forced, ≤~40k generation, up to 10 compaction cycles), the int4
lm_head costs ~0.45% perplexity and shows no compounding, no extra DNFs and no reasoning loops. The bf16-lm_head
variant was not better on any generation test, and it would cost `-2x` 22% of its max context. Quantized linear
attention costs ~0.1-0.2% perplexity, growing slightly with depth, which is consistent in direction with the claim
but far smaller. bf16 linear attention does not fit a 24 GB card with a usable KV cache anyway.

Not covered: contexts past ~64k in generation, many more compaction cycles, or the reporter's own suite. Sample
sizes (12 episodes, 40 problems) resolve large effects like the claimed 8%, not 1-2% differences.
