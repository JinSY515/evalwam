"""Merge per-difficulty valbench5s dirs into one flat 120-sample dir (ckpt1000 layout).

Order easy -> hard -> medium (what ckpt1000 used). Files are symlinked, not copied.
Usage: merge120.py <src_valbench5s_dir> <dst_dir> [<reference samples_meta.json>]
"""
import json, os, re, sys

src, dst = sys.argv[1], sys.argv[2]
ref = sys.argv[3] if len(sys.argv) > 3 else None
os.makedirs(dst, exist_ok=True)
meta, off = {}, 0
pat = re.compile(r"^sample_(\d+)(.*)$")
for diff in ["easy", "hard", "medium"]:
    d = os.path.join(src, diff)
    m = json.load(open(os.path.join(d, "samples_meta.json")))
    n = len(m)
    assert n == 40, f"{d}: {n} samples, expected 40 (inference incomplete?)"
    for f in os.listdir(d):
        mm = pat.match(f)
        if not mm:
            continue
        j = int(mm.group(1)) + off
        link = os.path.join(dst, f"sample_{j:02d}{mm.group(2)}")
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(os.path.join(d, f), link)
    for k, v in m.items():
        v = dict(v); v["seq_idx"] = int(k) + off
        meta[str(int(k) + off)] = v
    off += n
json.dump(meta, open(os.path.join(dst, "samples_meta.json"), "w"), indent=1)
print(f"merged {len(meta)} samples -> {dst}")
if ref:
    r = json.load(open(ref))
    key = lambda x: (x["seq_dir"], int(x["start_frame"]), int(x["num_frames"]), x.get("goal_frame"))
    bad = [k for k in r if key(r[k]) != key(meta[k])]
    print(f"vs reference: {len(r)} windows, mismatches={len(bad)} {bad[:5]}")
