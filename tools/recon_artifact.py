#!/usr/bin/env python3
"""Build the multi-sample 3D recon viewer as an Artifact bundle (a page + one gzip/base64 data file per sample).

Artifact limits: 16 MB per text file, 64 MB per publish, 256 MB per version -> a hand-picked set at full density
(7 cm voxels, 40 m), each sample gzip'd (~7-9 MB) and base64'd into data/<i>.txt; the page fetches the one picked
and inflates it with DecompressionStream. Page = recon_site.html adapted to the Artifact skeleton.

  python3 recon_artifact.py --results_dir <valbench5s_all120> --samples 103 78 ... --out_dir <dir> --page <page.html>
"""
import argparse, base64, csv, gzip, importlib.util, json, os
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("rw", os.path.join(HERE, "recon_web.py")); rw = importlib.util.module_from_spec(_s)
_s.loader.exec_module(rw)


def one(args):
    D, i, out, vox, md = args
    blob = rw.pack(*rw.build(D, i, max_depth=md, voxel_m=vox))
    txt = base64.b64encode(gzip.compress(blob, 9))
    open(f"{out}/data/{i}.txt", "wb").write(txt)
    return i, len(blob), len(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True); ap.add_argument("--samples", type=int, nargs="+", required=True)
    ap.add_argument("--star", type=int, nargs="*", default=[]); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--page", required=True); ap.add_argument("--label", default=""); ap.add_argument("--title", default="Valbench Recon Top 10")
    ap.add_argument("--voxel", type=float, default=0.07); ap.add_argument("--max_depth", type=float, default=40.0)
    ap.add_argument("--clip", type=float, default=15.0)
    a = ap.parse_args()
    D = a.results_dir.rstrip("/"); os.makedirs(f"{a.out_dir}/data", exist_ok=True)
    meta = json.load(open(f"{D}/samples_meta.json"))
    S = {r["sample"]: r for r in csv.DictReader(open(f"{D}/suite_metrics.csv"))}
    C = {r["sample"]: r for r in csv.DictReader(open(f"{D}/cam77.csv"))}
    f = lambda d, k: round(float(d[k]), 3) if d.get(k) not in (None, "") else None
    idx = []
    for rank, i in enumerate(a.samples, 1):
        m = meta[str(i)]; sd = m["seq_dir"].split("val_bench5s/")[-1]; s, c = S[f"sample_{i:02d}"], C[f"sample_{i:02d}"]
        P = np.load(m["seq_dir"] + "/poses.npy")[int(m["start_frame"]):int(m["start_frame"]) + 77]
        idx.append({"i": i, "rank": rank, "star": i in a.star, "seq": sd, "diff": sd.split("/")[0].replace("id_", ""),
                    "goal_success": f(s, "goal_da3_success"), "act_roterr": f(c, "act_roterr"), "coh_roterr": f(c, "coh_roterr"),
                    "da3_roterr": f(s, "da3_roterr"), "turn": round(float(abs((P[-1, 5] - P[0, 5] + 180) % 360 - 180)), 1)})
    with ProcessPoolExecutor(len(a.samples)) as ex:
        for i, raw, txt in ex.map(one, [(D, i, a.out_dir, a.voxel, a.max_depth) for i in a.samples]):
            print(f"sample {i:3d}: raw {raw / 1e6:.1f} MB -> data/{i}.txt {txt / 1e6:.1f} MB")
    h = open(os.path.join(HERE, "recon_site.html")).read()
    for old, new in [
        ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
         '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n', ""),
        ("<title>Valbench Recon Viewer</title>",
         f'<title>{a.title}</title>\n<meta name="claude:viewport-fit-sides" content="cover">'),
        ("</head>\n<body>\n", ""), ("</body>\n</html>\n", ""),
        ('fetch("samples.json").then(r => r.json()).then(ix => {',
         "Promise.resolve(" + json.dumps({"label": a.label, "clip": a.clip, "samples": idx}) + ").then(ix => {"),
        ('const r = await fetch("data/" + s.i + ".bin"); if (!r.ok) throw new Error("HTTP " + r.status);\n'
         '    const buf = await r.arrayBuffer(); if (my !== loadSeq) return;',
         'const r = await fetch("data/" + s.i + ".txt"); if (!r.ok) throw new Error("HTTP " + r.status);\n'
         '    const b64 = (await r.text()).trim(), bin = atob(b64), gz = new Uint8Array(bin.length);\n'
         '    for (let j = 0; j < bin.length; j++) gz[j] = bin.charCodeAt(j);\n'
         '    const buf = await new Response(new Blob([gz]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();\n'
         '    if (my !== loadSeq) return;'),
        ("o.textContent = `${s.i} · ${s.seq.split(\"/\").slice(1).join(\" \")}`;",
         "o.textContent = `#${s.rank}${s.star ? \" ★\" : \"\"} · ${s.i} · ${s.seq.split(\"/\").slice(1).join(\" \")}`;"),
        ('  if (s.coh_roterr != null) c.push(`<span class="chip">coherence RotErr ${s.coh_roterr.toFixed(1)}°</span>`);',
         '  if (s.coh_roterr != null) c.push(`<span class="chip">coherence RotErr ${s.coh_roterr.toFixed(1)}°</span>`);\n'
         '  if (s.da3_roterr != null) c.push(`<span class="chip">DA3 vs GT RotErr ${s.da3_roterr.toFixed(1)}°</span>`);\n'
         '  if (s.turn != null) c.push(`<span class="chip">GT turn ${s.turn.toFixed(0)}°</span>`);'),
        ('$("seq").textContent = s.seq.replaceAll("/", "__");',
         '$("seq").textContent = (s.star ? "★ " : "") + s.seq.replaceAll("/", "__");'),
    ]:
        assert old in h, old[:60]
        h = h.replace(old, new)
    open(a.page, "w").write(h)
    print(f"page -> {a.page} ({os.path.getsize(a.page) / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
