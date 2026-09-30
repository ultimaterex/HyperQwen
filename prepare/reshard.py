"""Repack a model dir's model-*.safetensors into ~MAX_GB shards (tensor order = index order), leaving
model_extra_tensors.safetensors and every non-shard file as-is (hardlinked). Writes a new dir and verifies every
tensor is byte-identical to the source (sha256 of the raw tensor bytes).
usage: python prepare/reshard.py <src_dir> <dst_dir> [--max-gb 5]"""
import hashlib, json, os, re, shutil, struct, sys
from collections import OrderedDict

src, dst = sys.argv[1].rstrip("/") + "/", sys.argv[2].rstrip("/") + "/"
MAX = int(float(sys.argv[sys.argv.index("--max-gb") + 1]) * 1e9) if "--max-gb" in sys.argv else int(5e9)
SHARD = re.compile(r"^model-\d{5}-of-\d{5}\.safetensors$")

def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return n, h

def raw(path, n, entry):
    a, b = entry["data_offsets"]
    with open(path, "rb") as f:
        f.seek(8 + n + a)
        return f.read(b - a)

idx = json.load(open(src + "model.safetensors.index.json"))
wm = idx["weight_map"]
shards = sorted(f for f in os.listdir(src) if SHARD.match(f))
# tensors in the order they appear across the source shards (== layer order for this export)
order = []
headers = {}
for s in shards:
    n, h = header(src + s)
    headers[s] = (n, h)
    order += [(k, s) for k in h if k != "__metadata__"]
meta = headers[shards[0]][1].get("__metadata__", {"format": "pt"})

# plan output shards by byte size
plan, cur, cur_size = [], [], 0
for k, s in order:
    a, b = headers[s][1][k]["data_offsets"]
    if cur and cur_size + (b - a) > MAX:
        plan.append(cur); cur, cur_size = [], 0
    cur.append((k, s)); cur_size += b - a
if cur:
    plan.append(cur)
N = len(plan)
os.makedirs(dst, exist_ok=True)

new_wm = {k: v for k, v in wm.items() if v == "model_extra_tensors.safetensors" or not SHARD.match(v)}
for i, group in enumerate(plan, 1):
    name = f"model-{i:05d}-of-{N:05d}.safetensors"
    # build header with sequential offsets, then stream bytes
    hdr, off = OrderedDict(), 0
    for k, s in group:
        e = headers[s][1][k]; a, b = e["data_offsets"]
        hdr[k] = {"dtype": e["dtype"], "shape": e["shape"], "data_offsets": [off, off + (b - a)]}
        off += b - a
    hdr["__metadata__"] = meta
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(dst + name, "wb") as f:
        f.write(struct.pack("<Q", len(hb))); f.write(hb)
        for k, s in group:
            f.write(raw(src + s, headers[s][0], headers[s][1][k]))
    for k, _ in group:
        new_wm[k] = name
    print(f"wrote {name}: {len(group)} tensors, {off / 1e9:.2f} GB", flush=True)

idx["weight_map"] = new_wm
idx.setdefault("metadata", {})["total_size"] = idx.get("metadata", {}).get("total_size")
json.dump(idx, open(dst + "model.safetensors.index.json", "w"), indent=2)
for f in os.listdir(src):
    if SHARD.match(f) or f == "model.safetensors.index.json" or os.path.exists(dst + f):
        continue
    os.link(src + f, dst + f)

# verify: same tensor set, byte-identical contents, index points at the file that holds each tensor
def digests(d, files):
    out = {}
    for s in files:
        n, h = header(d + s)
        for k, e in h.items():
            if k != "__metadata__":
                out[k] = (e["dtype"], tuple(e["shape"]), hashlib.sha256(raw(d + s, n, e)).hexdigest(), s)
    return out
old = digests(src, shards + ["model_extra_tensors.safetensors"])
new = digests(dst, [f"model-{i:05d}-of-{N:05d}.safetensors" for i in range(1, N + 1)] + ["model_extra_tensors.safetensors"])
assert set(old) == set(new), (len(old), len(new))
bad = [k for k in old if old[k][:3] != new[k][:3]]
misrouted = [k for k, v in new.items() if new_wm.get(k) != v[3]]
print(f"verify: {len(new)} tensors, content mismatches {len(bad)}, index mismatches {len(misrouted)}")
sys.exit(1 if bad or misrouted else 0)
