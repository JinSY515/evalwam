"""Checkpoint viewer: same window, every checkpoint side by side (original mp4s) + per-sample metrics.
Benchmarks: valbench5s (CARLA, 120) and the LookOut sets (real Aria, 51 / turn77 31). Extra runs (EXTRA_RUNS) are appended as ft<step>. Selectable in the page.

Serves straight from the results dir (no copies). Checkpoints are discovered on every /api/data call
(any ckpt*/<bench dir> with suite_summary.json), so newly evaluated checkpoints show up on reload.
GT clips are encoded once from <seq_dir>/images/*.png into ./cache (lazily, plus a background warm-up).

  python server.py --port 20100 [--run_dir <results_syn/RUN>]
Routes: /  /cmp (baseline comparison, see cmp.py)  /roll (long-rollout ranking + viz/recon, see roll.py)  /cherry (cherry-pick finder, see cherry.py)  /mypick (manual frame pick, see mypick.py)  /ablation (ablations vs ours, see ablation.py)  /vcons (direct vs residual L_vcons)  /api/benches  /api/data?bench=ID  /media/<bench>/<ckpt>/<file>  /gt/<bench>/<i>.mp4  /goal/<bench>/<i>.png
"""
import importlib, argparse, csv, glob, hashlib, json, math, os, re, statistics as st, subprocess, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from cmp import Cmp, CB
from pick import Pick
from roll import Roll
from cherry import Cherry
import mypick

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
FPS = 16
# id, results subdir, label, difficulty blocks (all120 index -> difficulty) or None
BENCHES = [
    ("valbench5s", "valbench5s_all120", "valbench5s · CARLA 120", {"easy": 0, "hard": 40, "medium": 80}),
    ("lookout_1c_s100", "lookout_1c_s100_all51_lh", "LookOut B · start 100 · Aria 51", None),   # _lh = L/R-corrected GT poses
    ("lookout_1c", "lookout_1c_all51_lh", "LookOut A · start 4 · Aria 51", None),
    ("lookout_turn77", "lookout_turn77/samples_lh", "LookOut turn77 · Aria 31", None),
    ("lookout_turn77v2", "lookout_turn77v2/samples_lh", "LookOut C · turn77 · Aria 82 (val set 35)", None),
    ("lookout_turn_cv", "lookout_turn_cv_all51", "LookOut turn_cv · Aria 51 (153f)", None),
]
BK = {b[0]: b for b in BENCHES}
# per-sample groups for benches without difficulty blocks: {bench: {sample index: (group, reason)}}; untagged -> "나머지"
TAG_FILE = "/root/project/egowam/results_syn/goalprog_mode_real6_nomr_hg0_grad0.5/multi/v2/viewer_tags.json"
def load_tags():
    try:
        return {b: {int(i): tuple(v) for i, v in d.items()} for b, d in json.load(open(TAG_FILE)).items()}
    except Exception:
        return {}
# other runs whose checkpoints are shown after the main run's, as "<prefix><step>" (e.g. ft400 = ft4000 run ckpt 400)
EXTRA_RUNS = [("ft", "_ft4000")]
# (key, file, column, label, dir: 1 higher better / -1 lower better, group)
MET = [
    ("psnr", "video_quality_metrics.csv", "psnr", "PSNR", 1, "Video"),
    ("ssim", "video_quality_metrics.csv", "ssim", "SSIM", 1, "Video"),
    ("lpips", "video_quality_metrics.csv", "lpips", "LPIPS", -1, "Video"),
    ("met3r", "met3r_metrics.csv", "met3r", "MEt3R", -1, "Video"),
    ("act_rot", "suite_metrics.csv", "act_roterr", "Action RotErr (deg)", -1, "Action"),
    ("act_ate", "suite_metrics.csv", "act_ate", "Action ATE (m)", -1, "Action"),
    ("act_rpe", "suite_metrics.csv", "act_rpe_rot", "Action RPE RotErr (deg)", -1, "Action"),
    ("coh_rot", "suite_metrics.csv", "coh_roterr", "Coherence cam RotErr (deg)", -1, "Coherence"),
    ("coh_ate", "suite_metrics.csv", "coh_ate", "Coherence cam ATE", -1, "Coherence"),
    ("d_absrel", "suite_metrics.csv", "depth_C_absrel", "Depth AbsRel", -1, "Coherence"),
    ("d_d1", "suite_metrics.csv", "depth_C_delta1", "Depth δ<1.25", 1, "Coherence"),
    ("goal_succ", "suite_metrics.csv", "goal_da3_success", "Goal pose success", 1, "Goal"),
    ("goal_lpips", "goal_image_metrics.csv", "goal_lpips", "Goal image LPIPS", -1, "Goal"),
    ("goal_dino", "goal_image_metrics.csv", "goal_dino_sim", "Goal image DINO", 1, "Goal"),
    ("da3_rot", "suite_metrics.csv", "da3_roterr", "DA3 cam RotErr (deg)", -1, "DA3 vs GT"),
    ("da3_trans", "suite_metrics.csv", "da3_transerr", "DA3 cam TransErr (m)", -1, "DA3 vs GT"),
]
MEDIA_RE = re.compile(r"^sample_\d{2,3}\.mp4$")
ATTN_REPORT = "/root/project/egowam/results_syn/attn_wam/report"
VRES_DIR = "/root/project/egowam/results_syn/goalprog_mode_real6_nomr_hg0_grad0.5/ckpta4000_7500/vcons_resid_viz"
VCONS_DIR = "/root/project/egowam/results_syn/goalprog_mode_real6_nomr_hg0_grad0.5_direct/vcons_cmp/page"
CTYPE = {".mp4": "video/mp4", ".png": "image/png", ".jpg": "image/jpeg", ".html": "text/html; charset=utf-8", ".json": "application/json"}


def num(x):
    try:
        v = float(x)
        return None if math.isnan(v) else round(v, 6)
    except (TypeError, ValueError):
        return None


def rows(p):
    return {r["sample"]: r for r in csv.DictReader(open(p))} if os.path.exists(p) else {}


class Data:
    def __init__(self, run_dir):
        self.R, self.lock, self.cache = run_dir.rstrip("/"), threading.Lock(), {}
        self.gt_locks, self.gt_lock = {}, threading.Lock()

    def ckpts(self, sub, extra=True):
        """main-run ckpts (int, sorted) then extra-run ckpts ("ft400", ...)."""
        def steps(R):
            return sorted(int(m.group(1)) for d in glob.glob(f"{R}/ckpt*/{sub}/suite_summary.json")
                          if (m := re.search(r"/ckpt(\d+)/" + re.escape(sub) + "/suite_summary.json$", d)))
        out = steps(self.R)
        if extra:
            for pre, suf in EXTRA_RUNS:
                out += [f"{pre}{n}" for n in steps(self.R + suf)]
        return out

    def ckdir(self, c):
        """results dir of checkpoint id c (int = main run, "<prefix><step>" = extra run)."""
        if isinstance(c, int) or str(c).isdigit():
            return f"{self.R}/ckpt{int(c)}"
        for pre, suf in EXTRA_RUNS:
            if str(c).startswith(pre) and str(c)[len(pre):].isdigit():
                return f"{self.R}{suf}/ckpt{int(str(c)[len(pre):])}"
        raise KeyError(c)

    def benches(self):
        return [{"id": b, "label": lab, "ckpts": self.ckpts(sub)} for b, sub, lab, _ in BENCHES]

    def get(self, bench):
        _, sub, _, blocks = BK[bench]
        cks = self.ckpts(sub)
        if not cks:
            return None
        key = tuple((c, max((os.path.getmtime(p) for p in glob.glob(f"{self.ckdir(c)}/{sub}/*.csv")), default=0)) for c in cks) \
            + ((os.path.getmtime(TAG_FILE) if os.path.exists(TAG_FILE) else 0),)
        with self.lock:
            hit = self.cache.get(bench)
            if hit and hit[0] == key:
                return hit[1]
        val = self.build(bench, sub, blocks, cks)
        with self.lock:
            self.cache[bench] = (key, val)
        return val

    def meta(self, bench):
        """samples_meta of the first checkpoint that has this bench (GT windows are identical across kept ckpts)."""
        _, sub, _, _ = BK[bench]
        cks = self.ckpts(sub)
        return json.load(open(f"{self.ckdir(cks[0])}/{sub}/samples_meta.json")) if cks else {}

    def build(self, bench, sub, blocks, cks):
        R = self.R
        ref = json.load(open(f"{self.ckdir(cks[0])}/{sub}/samples_meta.json"))
        n = len(ref)
        win = lambda m, i: (m[str(i)]["seq_dir"], int(m[str(i)]["start_frame"]), int(m[str(i)]["num_frames"]))
        good, skipped, per = [], [], {}
        for c in cks:
            D = f"{self.ckdir(c)}/{sub}"
            m2 = json.load(open(f"{D}/samples_meta.json"))
            if len(m2) != n or any(win(ref, i) != win(m2, i) for i in range(n)):
                skipped.append(c); continue
            good.append(c)
            cache, per[c] = {}, {}
            for i in range(n):
                s = f"sample_{i:02d}"
                per[c][i] = {k: num(cache.setdefault(fn, rows(f"{D}/{fn}")).get(s, {}).get(col)) for k, fn, col, *_ in MET}
        samples = []; tags = load_tags().get(bench)
        for i in range(n):
            info = ref[str(i)]
            sd = info["seq_dir"].rstrip("/").split("/")
            d = next(x for x, o in sorted(blocks.items(), key=lambda t: -t[1]) if i >= o) if blocks else (tags.get(i, ("나머지", None))[0] if tags else None)
            if blocks:   # CARLA: .../id_<diff>/<Town>_Opt/<Weather>/seq_XXXX
                place = [("맵", sd[-3].replace("_Opt", "")), ("날씨", sd[-2]), ("시퀀스", sd[-1])]
            else:        # Aria: .../aria/<task>/<uuid>
                place = [("task", sd[-2]), ("시퀀스", sd[-1][:8] + "…" + sd[-1][-4:])]
                if tags and tags.get(i, (None, None))[1]:
                    place.append(("제외 근거", tags[i][1]))
            samples.append({"i": i, "diff": d, "place": place, "start": int(info["start_frame"]),
                            "nf": int(info["num_frames"]), "goal": info.get("goal_frame"),
                            "m": {k: [per[c][i][k] for c in good] for k, *_ in MET}})
        ORDER = ["easy", "medium", "hard"]
        groups = ["all"] + (sorted(blocks, key=lambda k: ORDER.index(k) if k in ORDER else len(ORDER)) if blocks
                            else sorted({x[0] for x in tags.values()}) + ["나머지"] if tags else [])
        agg = {}
        for k, *_ in MET:
            agg[k] = {}
            for d in groups:
                vals = []
                for c in good:
                    v = [per[c][i][k] for i in range(n) if (d == "all" or samples[i]["diff"] == d) and per[c][i][k] is not None]
                    vals.append(round(st.mean(v), 6) if v else None)
                agg[k][d] = vals
        fvd = {"fvd16": [], "fvdall": []}   # fvdall = FVD over all frames (key fvd77 / fvd153 by clip length)
        for c in good:
            p = f"{self.ckdir(c)}/{sub}/fvd_musiq.json"
            j = json.load(open(p)) if os.path.exists(p) else {}
            fvd["fvd16"].append(num(j.get("fvd16"))); fvd["fvdall"].append(num(j.get(f"fvd{j.get('frames', 77)}")))
        return json.dumps({"run": os.path.basename(R), "bench": bench, "label": BK[bench][2], "ckpts": good, "skipped": skipped,
                           "groups": groups, "metrics": [{"key": k, "label": l, "dir": dr, "group": g} for k, _, _, l, dr, g in MET],
                           "samples": samples, "agg": agg, "fvd": fvd}, separators=(",", ":")).encode()

    # ---- GT clips from frames
    def gt_clip(self, bench, i):
        info = self.meta(bench).get(str(i))
        if not info:
            return None
        return self.clip(info["seq_dir"], int(info["start_frame"]), int(info["num_frames"]))

    def clip(self, seq, start, nf):
        name = hashlib.md5(f"{seq}|{start}|{nf}".encode()).hexdigest()[:16] + ".mp4"
        out = os.path.join(CACHE, name)
        if os.path.exists(out):
            return out
        with self.gt_lock:
            lk = self.gt_locks.setdefault(name, threading.Lock())
        with lk:
            if not os.path.exists(out):
                os.makedirs(CACHE, exist_ok=True)
                tmp = out + ".tmp.mp4"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-start_number", str(start),
                                "-i", f"{seq}/images/%06d.png", "-frames:v", str(nf), "-vf", "scale=512:512",
                                "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp],
                               check=True, timeout=120)
                os.replace(tmp, out)
        return out

    def goal_png(self, bench, i):
        info = self.meta(bench).get(str(i))
        if not info or info.get("goal_frame") is None:
            return None
        p = f"{info['seq_dir']}/images/{int(info['goal_frame']):06d}.png"
        return p if os.path.isfile(p) else None

    def warm(self):
        for b, *_ in BENCHES:
            try:
                self.get(b)
                for i in range(len(self.meta(b))):
                    self.gt_clip(b, i)
            except Exception as e:   # keep serving even if one bench is broken
                print(f"[warm] {b}: {e}", flush=True)
        print("[warm] done", flush=True)


def _ck(x):
    """cherry ckpt id: 7500 (int) or an averaged ckpt "a4000_7500" (results dir ckpta4000_7500)."""
    x = str(x)
    if x.isdigit():
        return int(x)
    if re.fullmatch(r"a\d+(_e?\d+)+|ema\d+", x):
        return x
    raise ValueError(x)


def make_handler(data, cmp, pick, roll, cherry=None):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def send_bytes(self, b, ctype, code=200, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-cache")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(b)

        def send_file(self, p):
            size = os.path.getsize(p)
            ctype = CTYPE.get(os.path.splitext(p)[1], "application/octet-stream")
            start, end, code = 0, size - 1, 200
            m = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range") or "")
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1)); end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
                else:
                    start = max(0, size - int(m.group(2)))
                if start > end or start >= size:
                    return self.send_bytes(b"", "text/plain", 416, {"Content-Range": f"bytes */{size}"})
                code = 206
            self.send_response(code)
            self.send_header("Content-Type", ctype); self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1)); self.send_header("Cache-Control", "max-age=300")
            if code == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return
            with open(p, "rb") as f:
                f.seek(start); left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk); left -= len(chunk)

        def do_GET(self):
            u = urlparse(self.path); path = u.path
            try:
                if path in ("/", "/index.html"):
                    return self.send_bytes(open(f"{HERE}/index.html", "rb").read(), CTYPE[".html"])
                if path == "/early":
                    import early; importlib.reload(early)
                    return self.send_bytes(early.page(), CTYPE[".html"])
                if path == "/ablation":   # ablation runs vs ours (ablation.py, reuses early's loaders)
                    import early, ablation; importlib.reload(early); importlib.reload(ablation)
                    return self.send_bytes(ablation.page(), CTYPE[".html"])
                if path == "/rebuttal":   # coherence-metric validation (rebuttal.py, reads results_syn/coherence_validation)
                    import rebuttal; importlib.reload(rebuttal)
                    return self.send_bytes(rebuttal.page(), CTYPE[".html"])
                if path == "/attn":      # attention study report (static, built by attn_wam_report.py)
                    p = f"{ATTN_REPORT}/index.html"
                    if os.path.exists(p):
                        return self.send_bytes(open(p, "rb").read(), CTYPE[".html"])
                m = re.match(r"^/attn/([\w\-]+(?:/[\w\-]+)*\.(?:png|jpg|mp4|json))$", path)
                if m and ".." not in m.group(1):
                    p = f"{ATTN_REPORT}/{m.group(1)}"
                    if os.path.isfile(p):
                        return self.send_file(p)
                if path in ("/vcons", "/vcons/"):   # direct vs residual L_vcons (static, built by tools/vcons_cmp/build_page.py)
                    p = f"{VCONS_DIR}/index.html"
                    if os.path.exists(p):
                        return self.send_bytes(open(p, "rb").read(), CTYPE[".html"])
                m = re.match(r"^/vcons/([\w.\-]+\.(?:png|mp4|jpg|html))$", path)
                if m and ".." not in m.group(1):
                    p = f"{VCONS_DIR}/{m.group(1)}"
                    if os.path.isfile(p):
                        return self.send_file(p)
                if path in ("/vres", "/vres/"):   # L_vcons residual maps (static, built by vcons_resid_sample.py)
                    p = f"{VRES_DIR}/index.html"
                    if os.path.exists(p):
                        return self.send_bytes(open(p, "rb").read(), CTYPE[".html"])
                m = re.match(r"^/vres/([\w.\-]+\.(?:png|mp4|json|csv))$", path)
                if m and ".." not in m.group(1):
                    p = f"{VRES_DIR}/{m.group(1)}"
                    if os.path.isfile(p):
                        return self.send_file(p)
                if path == "/pick":
                    return self.send_bytes(open(f"{HERE}/pick.html", "rb").read(), CTYPE[".html"])
                if path == "/api/pick/data":
                    return self.send_bytes(pick.data(), CTYPE[".json"])
                m = re.match(r"^/pickf/([\w.\-]+)/(\w+)/(\d+)\.jpg$", path)
                if m:
                    p = pick.frame(m.group(1), m.group(2), int(m.group(3)))
                    if p:
                        return self.send_file(p)
                m = re.match(r"^/pickv/([\w.\-]+)/(\w+)\.mp4$", path)
                if m:
                    p = pick.video(m.group(1), m.group(2))
                    if p:
                        return self.send_file(p)
                if path == "/cherrystyles.png":
                    p = f"{HERE}/cache/cherry/style_gallery.png"
                    if os.path.exists(p):
                        return self.send_file(p)
                if path == "/mypick":   # manual cherry-pick (mypick.py): any sample, hand-picked frames
                    return self.send_bytes(open(f"{HERE}/mypick.html", "rb").read(), CTYPE[".html"])
                if path == "/api/mypick/samples":
                    q = parse_qs(u.query); g = lambda k, d: q.get(k, [d])[0]
                    return self.send_bytes(mypick.samples(cherry, _ck(g("ck", "7500")), g("bench", "all")), CTYPE[".json"])
                if path == "/api/mypick/curve":
                    q = parse_qs(u.query); g = lambda k, d: q.get(k, [d])[0]
                    r = mypick.curve(cherry, _ck(g("ck", "7500")), g("bench", ""), g("name", ""), cmp)
                    if r:
                        return self.send_bytes(r, CTYPE[".json"])
                m = re.match(r"^/mypickv/(\d+|a[\de_]+|ema\d+)/(\w+)/([\w.\-]+)/(\w+)\.mp4$", path)
                if m:
                    p = mypick.video_path(cherry, cmp, _ck(m.group(1)), m.group(2), m.group(3), m.group(4))
                    if p:
                        return self.send_file(p)
                if path == "/cherry":
                    return self.send_bytes(open(f"{HERE}/cherry.html", "rb").read(), CTYPE[".html"])
                if path == "/api/cherry":
                    q = parse_qs(u.query); g = lambda k, d: q.get(k, [d])[0]
                    return self.send_bytes(cherry.candidates(_ck(g("ck", "7500")), g("bench", "all"), float(g("t", "0.3")), float(g("m", "0.08")),
                                                             int(g("l", "5")), g("psnr", "0") == "1", int(g("kmin", "0"))), CTYPE[".json"])
                if path == "/api/cherry/rank":
                    q = parse_qs(u.query); g = lambda k, d: q.get(k, [d])[0]
                    return self.send_bytes(cherry.rank(_ck(g("ck", "7500")), g("bench", "all"), int(g("n", "5")), int(g("gmin", "6")),
                                                       float(g("ymax", "35")), g("cut", "0") == "1", kmin=int(g("kmin", "0"))), CTYPE[".json"])
                if path == "/api/cherry/spread":
                    q = parse_qs(u.query); g = lambda k, d: q.get(k, [d])[0]
                    return self.send_bytes(cherry.spread(_ck(g("ck", "7500")), g("bench", "all"), float(g("t", "0.3")), float(g("m", "0.05")),
                                                         int(g("kbeat", "5")), int(g("n", "5")), int(g("gmin", "8")), g("psnr", "0") == "1",
                                                         int(g("kmin", "0")), float(g("dmin", "0")),
                                                         float(g("ymax", "35")), g("cut", "0") == "1", float(g("bmin", "0"))), CTYPE[".json"])
                m = re.match(r"^/cherrystrip/(\d+|a[\de_]+|ema\d+)/(\w+)/([\w.\-]+)/([\d_]+)\.jpg$", path)
                if m:
                    p = cherry.strip(_ck(m.group(1)), m.group(2), m.group(3), [int(x) for x in m.group(4).split("_")])
                    if p:
                        return self.send_file(p)
                m = re.match(r"^/cherryroute2/(\d+|a[\de_]+|ema\d+)/(\w+)/([\w.\-]+)/([\d_]+)\.png$", path)
                if m:
                    p = cherry.route_detail(_ck(m.group(1)), m.group(2), m.group(3), [int(x) for x in m.group(4).split("_")])
                    if p:
                        return self.send_file(p)
                m = re.match(r"^/cherryroute/(\d+|a[\de_]+|ema\d+)/(\w+)/([\w.\-]+)/([\d_]+)\.png$", path)
                if m:
                    p = cherry.route(_ck(m.group(1)), m.group(2), m.group(3), [int(x) for x in m.group(4).split("_")])
                    if p:
                        return self.send_file(p)
                if path == "/api/cherry/export":
                    q = parse_qs(u.query); g = lambda k, d: q.get(k, [d])[0]
                    r = cherry.export(_ck(g("ck", "7500")), g("bench", ""), g("name", ""), [int(x) for x in g("ks", "").split("_") if x.isdigit()])
                    return self.send_bytes(json.dumps({"ok": bool(r), "dir": r}).encode(), CTYPE[".json"], 200 if r else 400)
                m = re.match(r"^/cherryimg/(\d+|a[\de_]+|ema\d+)/(\w+)/([\w.\-]+)/(\d+)\.jpg$", path)
                if m:
                    p = cherry.thumb(_ck(m.group(1)), m.group(2), m.group(3), int(m.group(4)))
                    if p:
                        return self.send_file(p)
                if path == "/roll":
                    return self.send_bytes(open(f"{HERE}/roll.html", "rb").read(), CTYPE[".html"])
                if path == "/api/roll/export":
                    q = parse_qs(u.query); g = lambda k, d: q.get(k, [d])[0]
                    jj = g("j", "")
                    r = roll.export(g("set", ""), int(jj)) if jj.isdigit() else None
                    return self.send_bytes(json.dumps({"ok": bool(r), "dir": r}).encode(), CTYPE[".json"], 200 if r else 400)
                if path == "/api/roll/sets":
                    return self.send_bytes(json.dumps(roll.sets()).encode(), CTYPE[".json"])
                if path == "/api/roll/data":
                    v = roll.get(parse_qs(u.query).get("set", [""])[0])
                    return self.send_bytes(v, CTYPE[".json"]) if v else self.send_bytes(b"no such set", "text/plain", 404)
                m = re.match(r"^/rollf/(\w+)/(.+)$", path)
                if m:
                    p = roll.file(m.group(1), m.group(2))
                    if p:
                        return self.send_file(p)
                if path == "/cmp":
                    return self.send_bytes(open(f"{HERE}/cmp.html", "rb").read(), CTYPE[".html"])
                if path == "/api/cmp/benches":
                    return self.send_bytes(json.dumps(cmp.benches()).encode(), CTYPE[".json"])
                if path == "/api/cmp/data":
                    q = parse_qs(u.query)
                    b = q.get("bench", ["valbench5s"])[0]
                    v = cmp.get(b, int(q.get("ck", ["0"])[0])) if b in CB else None
                    return self.send_bytes(v, CTYPE[".json"]) if v else self.send_bytes(b"no such bench/ckpt", "text/plain", 404)
                m = re.match(r"^/cmpv/(\w+)/(\w+)/([\w.\-]+)/video\.mp4$", path)
                if m:
                    p = cmp.video(m.group(1), m.group(2), m.group(3))
                    if p:
                        return self.send_file(p)
                m = re.match(r"^/cmpgt/(\w+)/(\d+)\.mp4$", path)
                if m and m.group(1) in CB:
                    p = cmp.gt(m.group(1), int(m.group(2)))
                    if p:
                        return self.send_file(p)
                m = re.match(r"^/cmpgoal/(\w+)/(\d+)_(base|ours)\.png$", path)
                if m and m.group(1) in CB:
                    p = cmp.goal(m.group(1), int(m.group(2)), m.group(3))
                    if p:
                        return self.send_file(p)
                if path == "/api/benches":
                    return self.send_bytes(json.dumps(data.benches()).encode(), CTYPE[".json"])
                if path == "/api/data":
                    b = parse_qs(u.query).get("bench", ["valbench5s"])[0]
                    v = data.get(b) if b in BK else None
                    return self.send_bytes(v, CTYPE[".json"]) if v else self.send_bytes(b"no such bench", "text/plain", 404)
                m = re.match(r"^/media/(\w+)/(\w+)/([^/]+)$", path)
                if m and m.group(1) in BK and MEDIA_RE.match(m.group(3)):
                    p = f"{data.ckdir(m.group(2))}/{BK[m.group(1)][1]}/{m.group(3)}"
                    if os.path.isfile(p):
                        return self.send_file(p)
                m = re.match(r"^/(gt|goal)/(\w+)/(\d+)\.(mp4|png)$", path)
                if m and m.group(2) in BK:
                    p = data.gt_clip(m.group(2), int(m.group(3))) if m.group(1) == "gt" else data.goal_png(m.group(2), int(m.group(3)))
                    if p:
                        return self.send_file(p)
                self.send_bytes(b"not found", "text/plain", 404)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                try:
                    self.send_bytes(f"error: {e}".encode(), "text/plain", 500)
                except Exception:
                    pass

        do_HEAD = do_GET
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=20100)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--run_dir", default="/root/project/egowam/results_syn/goalprog_mode_real6_nomr_hg0_grad0.5")
    a = ap.parse_args()
    data = Data(a.run_dir)
    cmp = Cmp(data)
    pick = Pick(data, cmp)
    roll = Roll()
    def _warm():
        pick.warm(); data.warm(); cmp.warm()
    threading.Thread(target=_warm, daemon=True).start()   # metrics cache + GT clips before the first page load
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(data, cmp, pick, roll, Cherry(data)))
    srv.daemon_threads = True
    print(f"ckpt viewer on {a.host}:{a.port}  run={a.run_dir}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
