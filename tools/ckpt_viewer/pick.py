"""Cherry-pick page: a hand-picked list of valbench5s windows (pick_list.txt), ours (ckpt4000, plus any
"(ckptNNNN)" named on the line) vs the 7 baselines, frame by frame.

Every source is decoded once into JPEG frames under cache/pick/<name>/<src>/<k>.jpg, k = 0..78 = GT frame first+k
(ours has 77 frames, so k = 77, 78 do not exist for it). pick_list.txt is re-read on every /api/pick/data call:
one sample per line, "<name> [note]"; ":star:" in the note shows as a star, "(ckptNNNN)" adds that checkpoint.
"""
import json, os, re, subprocess, threading

from cmp import CMP_ROOT, MODELS, MK

HERE = os.path.dirname(os.path.abspath(__file__))
LIST = os.path.join(HERE, "pick_list.txt")
FR = os.path.join(HERE, "cache", "pick")
OURS_CK = 4000
SUB = "valbench5s_all120"
NF = 79
SRC_RE = re.compile(r"^(gt|ck\d+|[a-z0-9_]+)$")


class Pick:
    def __init__(self, data, cmp):
        self.D, self.C = data, cmp
        self.locks, self.lock = {}, threading.Lock()

    def entries(self):
        out = []
        for ln in open(LIST):
            ln = ln.strip()
            if not ln:
                continue
            name, _, note = ln.partition(" ")
            note = note.strip()
            star = ":star:" in note or "★" in note
            extra = [int(x) for x in re.findall(r"ckpt(\d+)", note)]
            note = re.sub(r":star:", "", note).strip()
            out.append({"name": name, "note": note, "star": star, "extra": extra})
        return out

    def meta(self, ck):
        m = json.load(open(f"{self.D.R}/ckpt{ck}/{SUB}/samples_meta.json"))
        return {"__".join(v["seq_dir"].rstrip("/").split("/")[-4:]): (int(k), v) for k, v in m.items()}

    def data(self):
        ents = self.entries()
        cmpj = json.loads(self.C.get("valbench5s", OURS_CK))
        by_name = {s["name"]: s for s in cmpj["samples"]}
        idx = self.meta(OURS_CK)
        out = []
        for n, e in enumerate(ents):
            if e["name"] not in idx:
                out.append({**e, "n": n, "missing": True}); continue
            i, info = idx[e["name"]]
            s = by_name.get(e["name"], {})
            cks = [OURS_CK] + [c for c in e["extra"] if c != OURS_CK and os.path.exists(f"{self.D.R}/ckpt{c}/{SUB}/samples_meta.json")]
            out.append({**e, "n": n, "i": i, "diff": s.get("diff"), "place": s.get("place"), "first": int(info["start_frame"]),
                        "ours_nf": int(info["num_frames"]), "goal": info.get("goal_frame"), "last": s.get("last"),
                        "cks": cks, "m": s.get("m", {})})
        return json.dumps({"ours_ck": OURS_CK, "nf": NF,
                           "models": [{"id": m, "label": l} for m, l in MODELS],
                           "metrics": cmpj["metrics"], "samples": out}, separators=(",", ":")).encode()

    # ---- frames
    def _src_input(self, name, src):
        """ffmpeg input args for one source of one sample, or None."""
        if src == "gt":
            idx = self.meta(OURS_CK)
            if name not in idx:
                return None
            info = idx[name][1]
            return ["-start_number", str(int(info["start_frame"])), "-i", f"{info['seq_dir']}/images/%06d.png", "-frames:v", str(NF)]
        m = re.match(r"^ck(\d+)$", src)
        if m:
            ck = int(m.group(1))
            idx = self.meta(ck)
            if name not in idx:
                return None
            p = f"{self.D.R}/ckpt{ck}/{SUB}/sample_{idx[name][0]:02d}.mp4"
            return ["-i", p] if os.path.exists(p) else None
        if src in MK:
            p = f"{CMP_ROOT}/{src}/valbench5s/{name}/video.mp4"
            return ["-i", p] if os.path.exists(p) else None
        return None

    def frames_dir(self, name, src):
        d = os.path.join(FR, name, src)
        if os.path.exists(os.path.join(d, ".done")):
            return d
        with self.lock:
            lk = self.locks.setdefault((name, src), threading.Lock())
        with lk:
            if os.path.exists(os.path.join(d, ".done")):
                return d
            inp = self._src_input(name, src)
            if not inp:
                return None
            os.makedirs(d, exist_ok=True)
            subprocess.run(["ffmpeg", "-v", "error", "-y", *inp, "-vf", "scale=512:-2", "-q:v", "2",
                            "-start_number", "0", os.path.join(d, "%02d.jpg")], check=True, timeout=180)
            open(os.path.join(d, ".done"), "w").close()
        return d

    def frame(self, name, src, k):
        if not re.match(r"^[\w.\-]+$", name) or not SRC_RE.match(src):
            return None
        d = self.frames_dir(name, src)
        if not d:
            return None
        p = os.path.join(d, f"{k:02d}.jpg")
        return p if os.path.isfile(p) else None

    def video(self, name, src):
        if src == "gt":
            idx = self.meta(OURS_CK)
            if name not in idx:
                return None
            info = idx[name][1]
            return self.D.clip(info["seq_dir"], int(info["start_frame"]), NF)
        inp = self._src_input(name, src) if re.match(r"^ck\d+$", src) or src in MK else None
        return inp[1] if inp else None

    def warm(self):
        try:
            for e in self.entries():
                srcs = ["gt", f"ck{OURS_CK}"] + [f"ck{c}" for c in e["extra"]] + [m for m, _ in MODELS]
                for s in srcs:
                    try:
                        self.frames_dir(e["name"], s)
                    except Exception as ex:
                        print(f"[pick warm] {e['name']} {s}: {ex}", flush=True)
        except Exception as ex:
            print(f"[pick warm] {ex}", flush=True)
        print("[pick warm] done", flush=True)
