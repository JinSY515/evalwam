"""Long-rollout page /roll: per-clip ranking of a multi-chunk rollout run + its visualisation / reconstruction.

A set = one merged rollout dir (rollout_NN.mp4, viz/rollout_NN_vs_gt.mp4 = GT|Rollout|Goal|Depth, viz/rollout_NN_quality.png,
recon/orbit_rollout_sNN_rgb.mp4, recon/bev_rollout_sNN.png, rollout_metrics.csv) plus the flat suite dir built from the
same rollouts (suite_metrics.csv, video_quality_metrics.csv, ...; samples_meta.json "src" = source rollout path).
The rollout dir's merge_map.json maps its index -> source rollout path (written by multi/rc4_vizrank.sh).

Ranking: per core metric, rank 1 = best over the clips that have it (ties averaged); composite = mean of those ranks.
Everything is re-read when a file changes (mtime key), so results appear as the pipeline writes them.
"""
import csv, json, math, os, re, sys, threading

RS = "/root/project/egowam/results_syn/goalprog_mode_real6_nomr_hg0_grad0.5"
# id, label, rollout (viz) dir, flat suite dirs (first one with suite_metrics.csv wins), recon subdir
SETS = [
    ("rc4_7500", "LookOut rc4 · ck7500 · 4 chunks = 305f · Aria 62",
     f"{RS}/ckpt7500/lookout_rc4_viz",
     [f"{RS}/ckpt7500/lookout_rc4_all62_lh", f"{RS}/ckpt7500/lookout_rc4_all62"], "recon"),
    ("rc4_a4000_7500", "LookOut rc4 · avg 4000+7500 · 4 chunks = 305f · Aria 62 (goal frames = ck7500)",
     f"{RS}/ckpta4000_7500/lookout_rc4_viz",
     [f"{RS}/ckpta4000_7500/lookout_rc4_all62_lh", f"{RS}/ckpta4000_7500/lookout_rc4_all62"], "recon"),
    ("rc4_a7500_8750", "LookOut rc4 · avg 7500+8750 · 4 chunks = 305f · Aria 62 (goal frames = ck7500)",
     f"{RS}/ckpta7500_8750/lookout_rc4_viz",
     [f"{RS}/ckpta7500_8750/lookout_rc4_all62_lh", f"{RS}/ckpta7500_8750/lookout_rc4_all62"], "recon"),
    ("rc4_a4000_7500_8750", "LookOut rc4 · avg 4000+7500+8750 · 4 chunks = 305f · Aria 62 (goal frames = ck7500)",
     f"{RS}/ckpta4000_7500_8750/lookout_rc4_viz",
     [f"{RS}/ckpta4000_7500_8750/lookout_rc4_all62_lh", f"{RS}/ckpta4000_7500_8750/lookout_rc4_all62"], "recon"),
    ("rc4_ema8000", "LookOut rc4 · EMA 8000 · 4 chunks = 305f · Aria 62 (goal frames = ck7500)",
     f"{RS}/ckptema8000/lookout_rc4_viz",
     [f"{RS}/ckptema8000/lookout_rc4_all62_lh", f"{RS}/ckptema8000/lookout_rc4_all62"], "recon"),
    ("rc4_a7500_e8000", "LookOut rc4 · avg 7500+EMA 8000 · 4 chunks = 305f · Aria 62 (goal frames = ck7500)",
     f"{RS}/ckpta7500_e8000/lookout_rc4_viz",
     [f"{RS}/ckpta7500_e8000/lookout_rc4_all62_lh", f"{RS}/ckpta7500_e8000/lookout_rc4_all62"], "recon"),
    ("rc4_ema8250", "LookOut rc4 · EMA 8250 · 4 chunks = 305f · Aria 62 (goal frames = ck7500)",
     f"{RS}/ckptema8250/lookout_rc4_viz",
     [f"{RS}/ckptema8250/lookout_rc4_all62_lh", f"{RS}/ckptema8250/lookout_rc4_all62"], "recon"),
    ("rc4_a7500_e8250", "LookOut rc4 · avg 7500+EMA 8250 · 4 chunks = 305f · Aria 62 (goal frames = ck7500)",
     f"{RS}/ckpta7500_e8250/lookout_rc4_viz",
     [f"{RS}/ckpta7500_e8250/lookout_rc4_all62_lh", f"{RS}/ckpta7500_e8250/lookout_rc4_all62"], "recon"),
]
SK = {s[0]: s for s in SETS}
# key, source (vq/s/ro/m3/gi), column, label, dir (+1 higher better, -1 lower better), group, core
RM = [
    ("psnr", "vq", "psnr", "PSNR", 1, "Video", True),
    ("ssim", "vq", "ssim", "SSIM", 1, "Video", False),
    ("lpips", "vq", "lpips", "LPIPS", -1, "Video", True),
    ("psnr_c1", "ro", "psnr_first_chunk", "PSNR chunk 1", 1, "Video", False),
    ("psnr_cl", "ro", "psnr_last_chunk", "PSNR last chunk", 1, "Video", False),
    ("met3r", "m3", "met3r", "MEt3R", -1, "Video", False),
    ("act_roterr", "c", "act_roterr", "Action RotErr (deg)", -1, "Action vs GT", True),
    ("act_ate_path", "c", "act_ate_path", "Action ATE / path", -1, "Action vs GT", True),
    ("act_rpe_rot", "c", "act_rpe_rot", "Action RPE Rot (deg)", -1, "Action vs GT", False),
    ("da3_roterr", "s", "da3_roterr", "DA3 cam RotErr (deg)", -1, "DA3 cam vs GT", True),
    ("da3_ate_path", "s", "da3_ate_path", "DA3 cam ATE / path", -1, "DA3 cam vs GT", True),
    ("coh_roterr", "c", "coh_roterr", "Coherence cam RotErr (deg)", -1, "Coherence", False),
    ("coh_ate_path", "c", "coh_ate_path", "Coherence cam ATE / path", -1, "Coherence", False),
    ("depth_delta1", "s", "depth_C_delta1", "Depth δ<1.25", 1, "Coherence", False),
    ("goal_act_pos", "s", "goal_act_final_pos_path", "Goal final pos / path (action)", -1, "Goal", True),
    ("goal_act_head", "s", "goal_act_final_head_deg", "Goal final heading (deg, action)", -1, "Goal", False),
    ("goal_act_success", "s", "goal_act_success", "Goal success (action)", 1, "Goal", False),
    ("goal_da3_success", "s", "goal_da3_success", "Goal success (DA3)", 1, "Goal", False),
    ("goal_lpips", "gi", "goal_lpips", "Goal image LPIPS", -1, "Goal", False),
    ("sim_rank", "rs", "sim_rank", "Recon/BEV sim vs GT (mean rank)", -1, "Recon vs GT", False),
    ("sim_bev_f1", "rs", "bev_f1", "BEV occupancy F1 (1 m)", 1, "Recon vs GT", False),
    ("sim_bev_iou", "rs", "bev_iou", "BEV occupancy IoU", 1, "Recon vs GT", False),
    ("sim_rec_f1", "rs", "rec_f1", "3D recon F1 (0.5 m)", 1, "Recon vs GT", False),
    ("sim_area_ratio", "rs", "area_ratio", "BEV area / GT", 1, "Recon vs GT", False),
    ("sim_area_err", "rs", "area_err", "BEV |log area ratio|", -1, "Recon vs GT", False),
    ("sim_traj_err", "rs", "traj_err", "BEV track gap / path", -1, "Recon vs GT", False),
]
# "rs" = <rollout dir>/recon_sim.csv (recon_sim.py): PRED d + PRED pose cloud vs GT d + GT pose, keyed like "ro"
# "c" = cam77.csv: camera rows in the external-table convention (all frames, RPE per frame), never mixed with eval_suite's
SRCF = {"vq": "video_quality_metrics.csv", "s": "suite_metrics.csv", "m3": "met3r_metrics.csv", "gi": "goal_image_metrics.csv",
        "c": "cam77.csv"}
FILE_RE = re.compile(r"^(rollout_\d+\.mp4|viz/rollout_\d+_(vs_gt\.mp4|quality\.png)|recon/(orbit_rollout_s\d+_rgb\.mp4|bev_rollout_s\d+\.png|grow_rollout_s\d+\.mp4))$")


def _num(x):
    if x in ("True", "False"):
        return 1.0 if x == "True" else 0.0
    try:
        v = float(x)
        return None if math.isnan(v) or math.isinf(v) else v
    except (TypeError, ValueError):
        return None


def _rows(p):
    if not os.path.exists(p):
        return {}
    out = {}
    for r in csv.DictReader(open(p)):
        k = r.get("sample", "")
        m = re.search(r"(\d+)$", k)
        if m:
            out[int(m.group(1))] = r
    return out


def _ranks(vals, d):
    """vals: {j: v} -> {j: rank} with 1 = best, ties averaged."""
    it = sorted(vals.items(), key=lambda t: -t[1] * d)
    out, i = {}, 0
    while i < len(it):
        k = i
        while k + 1 < len(it) and it[k + 1][1] == it[i][1]:
            k += 1
        for t in it[i:k + 1]:
            out[t[0]] = (i + k) / 2 + 1
        i = k + 1
    return out


class Roll:
    def __init__(self):
        self.cache, self.lock = {}, threading.Lock()

    def sets(self):
        return [{"id": s[0], "label": s[1], "short": s[1].split(" · ")[1], "ready": os.path.exists(f"{s[2]}/rollout_meta.json")}
                for s in SETS]

    def flat(self, sid):
        for d in SK[sid][3]:
            if os.path.exists(f"{d}/suite_metrics.csv"):
                return d
        return None

    def _key(self, sid):
        V, F = SK[sid][2], self.flat(sid)
        ps = [f"{V}/rollout_meta.json", f"{V}/rollout_metrics.csv", f"{V}/recon_sim.csv"] + ([f"{F}/{f}" for f in SRCF.values()] if F else [])
        mt = tuple(os.path.getmtime(p) if os.path.exists(p) else 0 for p in ps)
        nfiles = sum(len(os.listdir(f"{V}/{s}")) for s in ("viz", SK[sid][4]) if os.path.isdir(f"{V}/{s}"))
        return (sid, F, mt, nfiles)

    def get(self, sid):
        if sid not in SK:
            return None
        key = self._key(sid)
        with self.lock:
            if self.cache.get(sid, (None,))[0] == key:
                return self.cache[sid][1]
        b = self._build(sid)
        with self.lock:
            self.cache[sid] = (key, b)
        return b

    def _build(self, sid):
        _, label, V, _, rec = SK[sid]
        F = self.flat(sid)
        if not os.path.exists(f"{V}/rollout_meta.json"):
            return json.dumps({"id": sid, "label": label, "ready": False,
                               "msg": "rollout dir not merged yet (waiting for the rc4 suite)"}).encode()
        meta = json.load(open(f"{V}/rollout_meta.json"))
        mp = json.load(open(f"{V}/merge_map.json")) if os.path.exists(f"{V}/merge_map.json") else \
            {k: f"{V}/rollout_{int(k):02d}" for k in meta}
        src2j = {os.path.normpath(v): int(k) for k, v in mp.items()}
        # flat sample index -> viz index
        f2j = {}
        if F and os.path.exists(f"{F}/samples_meta.json"):
            for k, v in json.load(open(f"{F}/samples_meta.json")).items():
                s = os.path.normpath(os.path.realpath(v.get("src", ""))) if v.get("src") else None
                j = src2j.get(os.path.normpath(v.get("src", ""))) if v.get("src") else None
                if j is None and s:
                    j = next((jj for p, jj in src2j.items() if os.path.realpath(p) == s), None)
                if j is not None:
                    f2j[int(k)] = j
        src = {k: (_rows(f"{F}/{fn}") if F else {}) for k, fn in SRCF.items()}
        src["ro"] = _rows(f"{V}/rollout_metrics.csv")
        src["rs"] = _rows(f"{V}/recon_sim.csv")
        samples = []
        for k in sorted(meta, key=int):
            j = int(k); v = meta[k]
            fi = next((a for a, b in f2j.items() if b == j), None)
            m = {}
            for key, s, col, *_ in RM:
                r = src[s].get(j if s in ("ro", "rs") else fi, {}) if (s in ("ro", "rs") or fi is not None) else {}
                val = _num(r.get(col))
                if val is not None:
                    m[key] = val
            files = {
                "raw": f"rollout_{j:02d}.mp4", "vs_gt": f"viz/rollout_{j:02d}_vs_gt.mp4",
                "quality": f"viz/rollout_{j:02d}_quality.png",
                "orbit": f"{rec}/orbit_rollout_s{j:02d}_rgb.mp4", "bev": f"{rec}/bev_rollout_s{j:02d}.png",
                "grow": f"{rec}/grow_rollout_s{j:02d}.mp4"}
            files = {a: p for a, p in files.items() if os.path.exists(f"{V}/{p}") and os.path.getsize(f"{V}/{p}") > 0}
            seq = v["seq_dir"].rstrip("/")
            samples.append({"j": j, "name": os.path.basename(seq), "seq": seq, "start": v.get("start_frame"),
                            "nf": v.get("num_frames"), "chunks": v.get("num_chunks"), "goals": v.get("goal_frames"),
                            "flat": fi, "m": m, "files": files})
        # ranks + composite over core metrics
        ranks = {}
        for key, _, _, _, d, _, core in RM:
            vals = {s["j"]: s["m"][key] for s in samples if key in s["m"]}
            ranks[key] = _ranks(vals, d)
        core = [r[0] for r in RM if r[6]]
        for s in samples:
            s["r"] = {k: ranks[k][s["j"]] for k in ranks if s["j"] in ranks[k]}
            cr = [s["r"][k] for k in core if k in s["r"]]
            s["comp"] = sum(cr) / len(cr) if cr else None
            s["ncore"] = len(cr)
        order = sorted([s for s in samples if s["comp"] is not None], key=lambda s: s["comp"])
        for i, s in enumerate(order):
            s["rank"] = i + 1
        mean = {}
        for key, *_ in RM:
            v = [s["m"][key] for s in samples if key in s["m"]]
            if v:
                mean[key] = sum(v) / len(v)
        nviz = sum("vs_gt" in s["files"] for s in samples); nrec = sum("orbit" in s["files"] for s in samples)
        return json.dumps({"id": sid, "label": label, "ready": True, "flat": F,
                           "metrics": [{"key": k, "label": l, "dir": d, "group": g, "core": c} for k, _, _, l, d, g, c in RM],
                           "core": core, "mean": mean, "n": len(samples), "nviz": nviz, "nrecon": nrec,
                           "samples": samples}, separators=(",", ":")).encode()

    def file(self, sid, rel):
        if sid not in SK or not FILE_RE.match(rel):
            return None
        rel = rel.replace("recon/", SK[sid][4] + "/", 1) if rel.startswith("recon/") else rel
        p = f"{SK[sid][2]}/{rel}"
        return p if os.path.isfile(p) else None


EXPORT = f"{RS}/cherry_export"


def _export(self, sid, j):
    """/roll 저장: cherry_export{_ck<ck>}/lookout_rc4/<aria__nav__seq__fSSS-EEE>/ (same tree as /cherry)
      frames/{gt,ours}/fKKK.png  recon/{gt,ours}/azAAA.png  bev/{gt,ours}.png  meta.json"""
    if sid not in SK:
        return None
    V = SK[sid][2]
    meta = json.load(open(f"{V}/rollout_meta.json"))
    if str(j) not in meta:
        return None
    v = meta[str(j)]; ck = os.path.basename(os.path.dirname(V))[4:]          # ckpt7500 -> 7500, ckpta7500_8750 -> a7500_8750
    seq = v["seq_dir"].rstrip("/"); st, nf = int(v["start_frame"]), int(v["num_frames"])
    name = "__".join(seq.split("/")[-3:]) + f"__f{st:03d}-{st + nf - 1:03d}"
    root = f"{EXPORT}{'' if ck == '7500' else '_ck' + ck}/lookout_rc4/{name}"
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import recon_fig
    ks = [0, 76, 152, 228, 304]
    err = recon_fig.render(V, j, recon_fig.layout(root), ks)
    if err:
        return None
    sim = _rows(f"{V}/recon_sim.csv").get(j, {})
    json.dump({"set": sid, "ckpt": ck, "rollout_index": j, "seq_dir": seq, "start_frame": st, "num_frames": nf,
               "frames": ks, "gt_frames": [st + k for k in ks], "goal_frames": v.get("goal_frames"),
               "gt": "GT depth + GT pose (frames = GT images)", "ours": "PRED depth + PRED pose (frames = rollout mp4)",
               "recon": "orbit stills, elev 22, azimuth 0/90/180/270; GT and ours share one extent; same pixel mask for both: GT and PRED depth < 20 m, flying pixels (relative depth jump > 0.05) removed",
               "recon_sim": {k: round(float(x), 3) for k, x in sim.items() if k != "sample"}},
              open(f"{root}/meta.json", "w"), indent=1)
    return root


Roll.export = _export
