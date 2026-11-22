"""Baseline comparison: ours (one hg0 checkpoint) vs the 7 baselines of
dhyun22/wandering_inference_comparison_output, same GT window, per-sample metrics side by side.

Baseline folder layout: {model}/{valbench5s|lookout}/{sample}/{video.mp4, meta.json, suite_video.json, suite_pose_depth.json}
Samples are matched to ours by (last 4 parts of seq_dir, first frame).
Baseline windows are GT frames first..first+78 (79 f, goal = last frame); ours are start..start+76 (77 f, goal = start+81).
Frame k of every clip is GT frame first+k, so the page syncs by frame index.
"""
import csv, glob, json, os, re, statistics as st, subprocess, threading

CMP_ROOT = "/root/data/eval_bench/wandering_cmp"
MODELS = [   # paper table order
    ("wan21_flf2v", "Wan2.1-FLF2V"),
    ("wan21_vace", "Wan2.1-VACE"),
    ("wan22_fun_inp", "Wan2.2-Fun-InP"),
    ("vggtomega_seva", "VGGT-Ω+SEVA"),
    ("vggtomega_vggtdiff", "VGGT-Ω+VGGT-Diff"),
    ("nwm_nomad", "NWM"),
    ("nomad_deepverse", "DeepVerse"),
    ("nomad_lingbot_oneshot", "LingBot-World"),
    ("nomad_egowm", "EgoWM"),
    ("aether_planning", "Aether"),
]
MK = dict(MODELS)
# id, baseline folder, our results subdir, our viewer bench id (for /media), label, difficulty blocks
CBENCH = [
    ("valbench5s", "valbench5s", "valbench5s_all120", "valbench5s", "valbench5s · CARLA 120", {"easy": 0, "hard": 40, "medium": 80}),
    ("lookout_f100", "lookout", "lookout_1c_s100_all51", "lookout_1c_s100", "LookOut B · f100 · 46 (제외 5)", None),
    ("lookout_f004", "lookout", "lookout_1c_all51", "lookout_1c", "LookOut A · f004 · 39 (제외 12)", None),
    # C: turn77 val set (35 of the 82 turn77 clips); baselines uploaded 2026-11-14 into lookout/ (dataset dir lookout_turn77_val_cv,
    # same GT frames as our lookout_turn77_cv_lh -> matched by the last 3 path parts, see _tail)
    ("lookout_c", "lookout", "lookout_turn77v2/samples_lh", "lookout_turn77v2", "LookOut C · turn77 val set 35", None),
    # A+B+C combined (120 = 39 + 46 + 35): built from the three benches above, groups A/B/C
    ("lookout_abc", "lookout", None, None, "LookOut A+B+C · 120 (제외명단 제외)", None),
]
ABC = [("A", "lookout_f004"), ("B", "lookout_f100"), ("C", "lookout_c")]
# per-bench sample filters, re-read on every request: exclude = drop these indices, include = keep only these
V2 = "/root/project/egowam/results_syn/goalprog_mode_real6_nomr_hg0_grad0.5/multi/v2"
def sample_filter(bench):
    try:
        if bench in ("lookout_f100", "lookout_f004"):
            e = json.load(open(f"{V2}/lookout_AB_exclude17.json"))
            return "exclude", set(e["B" if bench == "lookout_f100" else "A"])
        if bench == "lookout_c":
            return "include", set(json.load(open(f"{V2}/turn77_union_list.json"))["idx82"])
    except Exception as ex:
        print(f"[cmp] filter {bench}: {ex}", flush=True)
    return None, set()
def filter_mtime():
    return tuple(os.path.getmtime(f"{V2}/{f}") if os.path.exists(f"{V2}/{f}") else 0 for f in ("lookout_AB_exclude17.json", "turn77_union_list.json"))
CB = {b[0]: b for b in CBENCH}
# key, (our file, our column), (baseline json, key), label, dir, group, core (used for the overall rank)
CMET = [
    ("psnr", ("video_quality_metrics.csv", "psnr"), ("suite_video.json", "fid_psnr"), "PSNR", 1, "Video", True),
    ("ssim", ("video_quality_metrics.csv", "ssim"), ("suite_video.json", "fid_ssim"), "SSIM", 1, "Video", False),
    ("lpips", ("video_quality_metrics.csv", "lpips"), ("suite_video.json", "fid_lpips"), "LPIPS", -1, "Video", True),
    ("met3r", ("met3r_metrics.csv", "met3r"), ("suite_video.json", "met3r"), "MEt3R", -1, "Video", False),
    ("goal_lpips", ("goal_image_metrics.csv", "goal_lpips"), ("suite_video.json", "goalimg_lpips"), "Goal image LPIPS", -1, "Goal", False),
    ("goal_dino", ("goal_image_metrics.csv", "goal_dino_sim"), ("suite_video.json", "goalimg_dino"), "Goal image DINO", 1, "Goal", False),
    ("goal_succ", ("suite_metrics.csv", "goal_da3_success"), ("suite_pose_depth.json", "goal_rgb_success"), "Goal success (DA3 cam)", 1, "Goal", True),
    ("goal_act", ("suite_metrics.csv", "goal_act_success"), ("suite_pose_depth.json", "goal_action_success"), "Goal success (action)", 1, "Goal", False),
    ("cam_rot", ("suite_metrics.csv", "da3_roterr"), ("suite_pose_depth.json", "rgbcam_abs_rot_deg"), "DA3 cam RotErr (deg)", -1, "Camera (DA3 vs GT)", True),
    ("cam_trans", ("suite_metrics.csv", "da3_transerr"), ("suite_pose_depth.json", "rgbcam_abs_trans"), "DA3 cam TransErr", -1, "Camera (DA3 vs GT)", False),
    ("cam_ate", ("suite_metrics.csv", "da3_ate"), ("suite_pose_depth.json", "rgbcam_abs_trans_rmse"), "DA3 cam ATE", -1, "Camera (DA3 vs GT)", True),
    ("act_rot", ("suite_metrics.csv", "act_roterr"), ("suite_pose_depth.json", "faith_abs_rot_deg"), "Action RotErr (deg)", -1, "Action vs GT", False),
    ("act_ate", ("suite_metrics.csv", "act_ate"), ("suite_pose_depth.json", "faith_abs_trans_rmse"), "Action ATE", -1, "Action vs GT", False),
    ("coh_rot", ("suite_metrics.csv", "coh_roterr"), ("suite_pose_depth.json", "coh_cam_abs_rot_deg"), "Coherence RotErr (deg)", -1, "Coherence", False),
    # coherence ATE: ours is in metres (DA3 ref already sim3-fit to GT), baselines' DA3 ref is raw DA3 units
    # -> compare scale-free: ATE / DA3 reference path length (ours coh_ate_path; baselines rmse / coh_cam_ref_traj_len)
    ("coh_ate", ("suite_metrics.csv", "coh_ate_path"), ("suite_pose_depth.json", ("coh_cam_abs_trans_rmse", "coh_cam_ref_traj_len")), "Coherence cam ATE / path", -1, "Coherence", False),
    ("d_absrel", ("suite_metrics.csv", "depth_C_absrel"), ("suite_pose_depth.json", "coh_depth_absrel"), "Depth AbsRel", -1, "Coherence", False),
    ("d_d1", ("suite_metrics.csv", "depth_C_delta1"), ("suite_pose_depth.json", "coh_depth_delta_1.25"), "Depth δ<1.25", 1, "Coherence", False),
]
CORE = [k for k, *_, core in CMET if core]
NAME_RE = re.compile(r"^[\w.\-]+$")


def _num(x):
    try:
        v = float(x)
        return None if v != v else round(v, 6)
    except (TypeError, ValueError):
        return None


def _val(j, bk):
    """baseline value; a (num, den) tuple means num / den (scale-free ratio)."""
    if isinstance(bk, tuple):
        a, b = _num(j.get(bk[0])), _num(j.get(bk[1]))
        return round(a / b, 6) if a is not None and b else None
    return _num(j.get(bk))


def _tail(seq):
    # Aria: aria/nav/<seq> (the dataset root differs between ours and the baselines for LookOut C); CARLA: diff/town/weather/seq
    p = seq.rstrip("/").split("/")
    return "/".join(p[-3:] if "aria" in p[-3:-1] else p[-4:])


class Cmp:
    def __init__(self, data):
        self.D, self.lock, self.cache, self.fps = data, threading.Lock(), {}, {}

    # ---- baseline index: (tail(seq_dir), first_idx) -> sample folder name, per baseline folder
    def index(self, folder):
        key = ("idx", folder)
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        idx = {}
        ref = MODELS[0][0]
        for p in glob.glob(f"{CMP_ROOT}/{ref}/{folder}/*/meta.json"):
            j = json.load(open(p))
            idx[(_tail(j["seq_dir"]), int(j["first_idx"]))] = (os.path.basename(os.path.dirname(p)), int(j["last_idx"]))
        with self.lock:
            self.cache[key] = idx
        return idx

    def model_fps(self, m, folder):
        k = (m, folder)
        if k not in self.fps:
            v = sorted(glob.glob(f"{CMP_ROOT}/{m}/{folder}/*/video.mp4"))
            fps = 16.0
            if v:
                try:
                    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                                        "-of", "csv=p=0", v[0]], capture_output=True, text=True, timeout=20).stdout.strip()
                    a, b = r.split("/")
                    fps = float(a) / float(b)
                except Exception:
                    pass
            self.fps[k] = fps
        return self.fps[k]

    def ours_ckpts(self, sub):
        return self.D.ckpts(sub, extra=False)

    def ours_rows(self, ck, sub):
        D = f"{self.D.R}/ckpt{ck}/{sub}"
        files = {}
        for _, (fn, _), *_ in CMET:
            if fn not in files:
                p = f"{D}/{fn}"
                files[fn] = {r["sample"]: r for r in csv.DictReader(open(p))} if os.path.exists(p) else {}
        return files

    def ours_metric(self, files, i, fn, col):
        return _num(files.get(fn, {}).get(f"sample_{i:02d}", {}).get(col))

    def best_ckpt(self, sub, cks):
        """checkpoint with the best mean per-metric rank over CORE metrics (bench mean)."""
        if len(cks) < 2:
            return cks[-1] if cks else None
        means = {}
        for c in cks:
            f = self.ours_rows(c, sub)
            means[c] = {}
            for k, (fn, col), _, _, dr, _, _ in CMET:
                if k not in CORE:
                    continue
                v = [_num(r.get(col)) for r in f.get(fn, {}).values()]
                v = [x for x in v if x is not None]
                means[c][k] = st.mean(v) * dr if v else None
        rank = {c: [] for c in cks}
        for k in CORE:
            vals = sorted((means[c][k] for c in cks if means[c][k] is not None), reverse=True)
            for c in cks:
                if means[c][k] is not None:
                    rank[c].append(vals.index(means[c][k]) + 1)
        return min(cks, key=lambda c: (st.mean(rank[c]) if rank[c] else 99, -c))

    def abc_ckpts(self):
        sets = [set(self.ours_ckpts(CB[sb][2])) for _, sb in ABC]
        return sorted(set.intersection(*sets))

    def get_abc(self, ck):
        """A+B+C in one list: samples re-indexed 0..N-1, keeping their source bench (src), viewer bench (vb) and index (vi)."""
        samples, agg_src, models, metrics = [], [], None, None
        for tag, sb in ABC:
            raw = self.get(sb, ck)
            if raw is None:
                return None
            d = json.loads(raw)
            if models is None or len(d["models"]) > len(models):
                models = d["models"]
            metrics = d["metrics"]
            for x in d["samples"]:
                y = dict(x, vb=d["viewer_bench"], vi=x["i"], src=sb, diff=tag, i=len(samples))
                if not CB[sb][1]:
                    y["nobase"] = True
                samples.append(y)
        n = len(models)
        for y in samples:
            y["m"] = {k: (row + [None] * (n - len(row))) for k, row in y["m"].items()}
        groups = ["all"] + [t for t, _ in ABC]
        agg = {}
        for k, *_ in CMET:
            agg[k] = {}
            for g in groups:
                cols = []
                for j in range(n):
                    v = [s["m"][k][j] for s in samples if (g == "all" or s["diff"] == g) and s["m"][k][j] is not None]
                    cols.append(round(st.mean(v), 6) if v else None)
                agg[k][g] = cols
        return json.dumps({"bench": "lookout_abc", "label": CB["lookout_abc"][4], "ck": ck, "viewer_bench": None, "groups": groups,
                           "models": models, "metrics": metrics, "samples": samples, "missing": [], "agg": agg,
                           "filter": {"mode": "A+B+C"}}, separators=(",", ":")).encode()

    def benches(self):
        out = []
        for b, folder, sub, vb, lab, _ in CBENCH:
            if b == "lookout_abc":
                cks = self.abc_ckpts(); out.append({"id": b, "label": lab, "ckpts": cks, "best": cks[-1] if cks else None}); continue
            cks = self.ours_ckpts(sub)
            out.append({"id": b, "label": lab, "ckpts": cks, "best": self.best_ckpt(sub, cks) if cks else None})
        return out

    def get(self, bench, ck):
        if bench == "lookout_abc":
            return self.get_abc(ck) if ck in self.abc_ckpts() else None
        b, folder, sub, vb, lab, blocks = CB[bench]
        cks = self.ours_ckpts(sub)
        if ck not in cks:
            return None
        files = self.ours_rows(ck, sub)
        mt = max((os.path.getmtime(p) for p in glob.glob(f"{self.D.R}/ckpt{ck}/{sub}/*.csv")), default=0)
        key = (bench, ck, mt, filter_mtime())
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        meta = json.load(open(f"{self.D.R}/ckpt{ck}/{sub}/samples_meta.json"))
        idx = self.index(folder) if folder else {}
        models = MODELS if folder else []
        fmode, fset = sample_filter(bench)
        bjson = {}
        samples, missing = [], []
        for i in range(len(meta)):
            if (fmode == "exclude" and i in fset) or (fmode == "include" and i not in fset):
                continue
            info = meta[str(i)]
            seq, start = info["seq_dir"], int(info["start_frame"])
            hit = idx.get((_tail(seq), start)) if folder else (f"c{i:02d}", start + int(info["num_frames"]) - 1)
            if not hit:
                missing.append(i); continue
            name, last = hit
            sd = seq.rstrip("/").split("/")
            d = next(x for x, o in sorted(blocks.items(), key=lambda t: -t[1]) if i >= o) if blocks else None
            if blocks:
                place = [("맵", sd[-3].replace("_Opt", "")), ("날씨", sd[-2]), ("시퀀스", sd[-1])]
            else:
                place = [("task", sd[-2]), ("시퀀스", sd[-1][:8] + "…" + sd[-1][-4:])]
            m = {}
            for k, (fn, col), (bf, bk), *_ in CMET:
                row = [self.ours_metric(files, i, fn, col)]
                for mid, _ in models:
                    jk = (mid, name, bf)
                    if jk not in bjson:
                        p = f"{CMP_ROOT}/{mid}/{folder}/{name}/{bf}"
                        bjson[jk] = json.load(open(p)) if os.path.exists(p) else {}
                    row.append(_val(bjson[jk], bk))
                m[k] = row
            samples.append({"i": i, "name": name, "diff": d, "place": place, "start": start, "nf": int(info["num_frames"]),
                            "goal": info.get("goal_frame"), "first": start, "last": last, "m": m})
        ORDER = ["easy", "medium", "hard"]
        groups = ["all"] + (sorted(blocks, key=lambda k: ORDER.index(k)) if blocks else [])
        agg = {}
        for k, *_ in CMET:
            agg[k] = {}
            for g in groups:
                cols = []
                for j in range(len(models) + 1):
                    v = [s["m"][k][j] for s in samples if (g == "all" or s["diff"] == g) and s["m"][k][j] is not None]
                    cols.append(round(st.mean(v), 6) if v else None)
                agg[k][g] = cols
        out = json.dumps({
            "bench": bench, "label": lab, "ck": ck, "viewer_bench": vb, "groups": groups,
            "models": [{"id": "ours", "label": f"Ours ck{ck}"}] + [{"id": mid, "label": l, "fps": self.model_fps(mid, folder)} for mid, l in models],
            "filter": {"mode": fmode, "n": len(fset)},
            "metrics": [{"key": k, "label": l, "dir": dr, "group": g, "core": c} for k, _, _, l, dr, g, c in CMET],
            "samples": samples, "missing": missing, "agg": agg}, separators=(",", ":")).encode()
        with self.lock:
            self.cache[key] = out
        return out

    # ---- media paths
    def video(self, model, bench, name):
        if model not in MK or bench not in CB or not CB[bench][1] or not NAME_RE.match(name):
            return None
        p = f"{CMP_ROOT}/{model}/{CB[bench][1]}/{name}/video.mp4"
        return p if os.path.isfile(p) else None

    def _sample(self, bench, i):
        b, folder, sub, *_ = CB[bench]
        cks = self.ours_ckpts(sub)
        if not cks:
            return None
        info = json.load(open(f"{self.D.R}/ckpt{cks[0]}/{sub}/samples_meta.json")).get(str(i))
        if not info:
            return None
        if not folder:
            return info, (f"c{i:02d}", int(info["start_frame"]) + int(info["num_frames"]) - 1)
        hit = self.index(folder).get((_tail(info["seq_dir"]), int(info["start_frame"])))
        return (info, hit) if hit else None

    def gt(self, bench, i):
        s = self._sample(bench, i)
        if not s:
            return None
        info, (name, last) = s
        start = int(info["start_frame"])
        return self.D.clip(info["seq_dir"], start, last - start + 1)

    def goal(self, bench, i, which):
        s = self._sample(bench, i)
        if not s:
            return None
        info, (name, last) = s
        f = last if which == "base" else info.get("goal_frame")
        if f is None:
            return None
        p = f"{info['seq_dir']}/images/{int(f):06d}.png"
        return p if os.path.isfile(p) else None

    def warm(self):
        for b, folder, sub, *_ in CBENCH:
            if sub is None:
                continue
            try:
                cks = self.ours_ckpts(sub)
                if not cks:
                    continue
                n = len(json.load(open(f"{self.D.R}/ckpt{cks[0]}/{sub}/samples_meta.json")))
                for i in range(n):
                    self.gt(b, i)
                for mid, _ in (MODELS if folder else []):
                    self.model_fps(mid, folder)
            except Exception as e:
                print(f"[cmp warm] {b}: {e}", flush=True)
        print("[cmp warm] done", flush=True)
