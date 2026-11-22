"""Cherry-pick finder: frames where ours is close to GT and clearly better than every baseline.

Data: cache/cherry/ck<ck>/<bench>/<name>.npz from cherry_compute.py (per-frame LPIPS / PSNR vs GT on the GT rows all
baselines cover, rows = ours + MODELS). A frame is "good" when ours LPIPS <= t and (min baseline LPIPS - ours) >= m
(optionally ours PSNR > every baseline); runs of >= l good frames are candidates, scored by mean margin * sqrt(length).
Routes: /cherry  /api/cherry?ck=&bench=&t=&m=&l=&psnr=  /cherryimg/<ck>/<bench>/<name>/<k>.jpg (GT | ours | best | worst baseline)
"""
import glob, json, math, os, threading
import numpy as np, cv2

from cmp import CMP_ROOT, MODELS, CB
from cherry_compute import cut, NF, W, H, sub_for

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "cache", "cherry")
BENCHES = ["valbench5s", "lookout_f004", "lookout_f100", "lookout_c"]
LABEL = {"valbench5s": "valbench5s", "lookout_f004": "LookOut A", "lookout_f100": "LookOut B", "lookout_c": "LookOut C"}


class Cherry:
    def __init__(self, data):
        self.D, self.lock, self.cache = data, threading.Lock(), {}

    def load(self, ck, bench):
        files = sorted(glob.glob(f"{ROOT}/ck{ck}/{bench}/*.npz"))
        key = (ck, bench, len(files), max((os.path.getmtime(f) for f in files), default=0))
        with self.lock:
            if self.cache.get((ck, bench), (None,))[0] == key:
                return self.cache[(ck, bench)][1]
        out = []
        for f in files:
            z = np.load(f)
            out.append({"name": os.path.basename(f)[:-4], "i": int(z["i"]), "start": int(z["start"]), "lp": z["lpips"], "ps": z["psnr"]})
        with self.lock:
            self.cache[(ck, bench)] = (key, out)
        return out

    def candidates(self, ck, bench, t, m, l, need_psnr, kmin=0):
        benches = BENCHES if bench == "all" else [bench]
        segs, nsamp = [], 0
        for b in benches:
            for s in self.load(ck, b):
                nsamp += 1
                lp, ps = s["lp"], s["ps"]
                o, base = lp[0], lp[1:]
                best = np.nanmin(base, axis=0); bestj = np.nanargmin(np.where(np.isnan(base), np.inf, base), axis=0)
                worstj = np.nanargmax(np.where(np.isnan(base), -np.inf, base), axis=0)
                margin = best - o
                good = (o <= t) & (margin >= m)
                if need_psnr:
                    good &= ps[0] > np.nanmax(ps[1:], axis=0)
                good &= np.isfinite(o) & np.isfinite(best)
                good[:kmin] = False   # skip the first frames (still close to the conditioning history)
                k = 0
                while k < NF:
                    if not good[k]:
                        k += 1; continue
                    e = k
                    while e + 1 < NF and good[e + 1]:
                        e += 1
                    if e - k + 1 >= l:
                        r = np.arange(k, e + 1); pk = int(r[np.argmax(margin[r])])
                        segs.append({"bench": b, "label": LABEL[b], "i": s["i"], "name": s["name"], "k0": int(k), "k1": int(e),
                                     "len": int(e - k + 1), "peak": pk, "gt_frame": s["start"] + pk,
                                     "ours_lp": round(float(o[pk]), 4), "best_lp": round(float(best[pk]), 4),
                                     "best_model": MODELS[int(bestj[pk])][1], "worst_model": MODELS[int(worstj[pk])][1],
                                     "margin": round(float(margin[r].mean()), 4), "peak_margin": round(float(margin[pk]), 4),
                                     "ours_ps": round(float(ps[0][pk]), 2), "best_ps": round(float(np.nanmax(ps[1:, pk])), 2),
                                     "sample_ours_lp": round(float(np.nanmean(o)), 4), "sample_best_lp": round(float(np.nanmean(np.nanmin(base, 0))), 4),
                                     "sample_win": round(float(np.mean(np.isfinite(o) & (o < best))), 3),
                                     "score": round(float(margin[r].mean() * math.sqrt(e - k + 1)), 4)})
                    k = e + 1
        segs.sort(key=lambda x: -x["score"])
        return json.dumps({"ck": ck, "bench": bench, "params": {"t": t, "m": m, "l": l, "psnr": need_psnr, "kmin": kmin}, "n_samples": nsamp,
                           "models": [l for _, l in MODELS], "segments": segs}, separators=(",", ":")).encode()

    def frame(self, p, k, crop):
        cap = cv2.VideoCapture(p); cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = cap.read()
        return cut(f, crop) if ok else np.zeros((H, W, 3), np.uint8)

    def thumb(self, ck, bench, name, k):
        if bench not in BENCHES or not all(c.isalnum() or c in "_.-" for c in name) or not (0 <= k < NF):
            return None
        out = f"{ROOT}/thumbs/ck{ck}/{bench}/{name}_{k:02d}.jpg"
        if os.path.exists(out):
            return out
        s = next((x for x in self.load(ck, bench) if x["name"] == name), None)
        if s is None:
            return None
        _, folder, sub, *_ = sub_for(ck, bench)
        info = json.load(open(f"{self.D.R}/ckpt{ck}/{sub}/samples_meta.json"))[str(s["i"])]
        gt = cut(cv2.imread(f"{info['seq_dir'].rstrip('/')}/images/{s['start'] + k:06d}.png"), None)
        ours = self.frame(f"{self.D.R}/ckpt{ck}/{sub}/sample_{s['i']:02d}.mp4", k, None)
        base = s["lp"][1:, k]; bj = int(np.nanargmin(np.where(np.isnan(base), np.inf, base))); wj = int(np.nanargmax(np.where(np.isnan(base), -np.inf, base)))
        def bf(j):
            mid = MODELS[j][0]; mp = f"{CMP_ROOT}/{mid}/{folder}/{name}"
            return self.frame(f"{mp}/video.mp4", k, json.load(open(f"{mp}/meta.json")).get("gt_crop_box_xyxy"))
        panels = [("GT", gt), (f"Ours ck{ck} {s['lp'][0, k]:.3f}", ours), (f"best: {MODELS[bj][1]} {base[bj]:.3f}", bf(bj)),
                  (f"worst: {MODELS[wj][1]} {base[wj]:.3f}", bf(wj))]
        tiles = []
        for lab, im in panels:
            im = im.copy(); cv2.rectangle(im, (0, 0), (W, 22), (20, 20, 20), -1)
            cv2.putText(im, lab, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA); tiles.append(im)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cv2.imwrite(out, np.concatenate(tiles, 1), [cv2.IMWRITE_JPEG_QUALITY, 85])
        return out


def spread_pick(Q, n):
    """n frames from sorted Q with the largest possible minimum gap (binary search on the gap + greedy chain,
    then n picks spread evenly along the chain). -> (frames, min gap) or (None, 0)."""
    if len(Q) < n:
        return None, 0
    lo, hi, best = 1, Q[-1] - Q[0], None
    while lo <= hi:
        g = (lo + hi) // 2; sel = [Q[0]]
        for q in Q[1:]:
            if q - sel[-1] >= g:
                sel.append(q)
        if len(sel) >= n:
            best, lo = sel, g + 1
        else:
            hi = g - 1
    if best is None:
        return Q[:n], 1
    idx = np.linspace(0, len(best) - 1, n).round().astype(int)
    s = [int(best[i]) for i in idx]
    return s, int(min(np.diff(s))) if n > 1 else 77


def _spread(self, ck, bench, t, m, kbeat, n, gmin, need_psnr, kmin):
    """per sequence: frames where ours LPIPS <= t and ours beats >= kbeat of the 7 baselines by >= m LPIPS
    (optionally higher PSNR than all), from frame kmin on; pick n of them with the largest minimum gap (>= gmin)."""
    benches = BENCHES if bench == "all" else [bench]
    rows, nsamp = [], 0
    for b in benches:
        for s in self.load(ck, b):
            nsamp += 1
            lp, ps = s["lp"], s["ps"]; o, base = lp[0], lp[1:]
            nbeat = np.sum((base - o[None]) >= m, axis=0)
            ok = np.isfinite(o) & (o <= t) & (nbeat >= kbeat)
            if need_psnr:
                ok &= ps[0] > np.nanmax(ps[1:], axis=0)
            ok[:kmin] = False
            Q = [int(k) for k in np.nonzero(ok)[0]]
            sel, g = spread_pick(Q, n)
            if sel is None or g < gmin:
                continue
            med = np.nanmedian(base, axis=0); bestv = np.nanmin(base, axis=0)
            fr = [{"k": k, "gt_frame": s["start"] + k, "ours_lp": round(float(o[k]), 4), "med_lp": round(float(med[k]), 4),
                   "best_lp": round(float(bestv[k]), 4), "best_model": MODELS[int(np.nanargmin(np.where(np.isnan(base[:, k]), np.inf, base[:, k])))][1],
                   "nbeat": int(nbeat[k]), "ours_ps": round(float(ps[0][k]), 2)} for k in sel]
            rows.append({"bench": b, "label": LABEL[b], "i": s["i"], "name": s["name"], "frames": fr, "n_ok": len(Q), "min_gap": g,
                         "span": sel[-1] - sel[0], "ours_lp": round(float(np.mean([f["ours_lp"] for f in fr])), 4),
                         "margin_med": round(float(np.mean([f["med_lp"] - f["ours_lp"] for f in fr])), 4),
                         "margin_best": round(float(np.mean([f["best_lp"] - f["ours_lp"] for f in fr])), 4),
                         "sample_ours_lp": round(float(np.nanmean(o)), 4)})
    rows.sort(key=lambda r: (-r["min_gap"], -r["margin_med"]))
    return json.dumps({"ck": ck, "bench": bench, "n_samples": nsamp, "params": {"t": t, "m": m, "kbeat": kbeat, "n": n, "gmin": gmin, "psnr": need_psnr, "kmin": kmin},
                       "rows": rows}, separators=(",", ":")).encode()


def _strip(self, ck, bench, name, ks):
    """frames ks side by side (columns); rows GT / ours / each of the 7 baselines at the same frame (labelled with LPIPS)."""
    if bench not in BENCHES or not all(c.isalnum() or c in "_.-" for c in name) or not ks or len(ks) > 12 or not all(0 <= k < NF for k in ks):
        return None
    out = f"{ROOT}/strips_all/ck{ck}/{bench}/{name}_{'-'.join(map(str, ks))}.jpg"
    if os.path.exists(out):
        return out
    s = next((x for x in self.load(ck, bench) if x["name"] == name), None)
    if s is None:
        return None
    _, folder, sub, *_ = sub_for(ck, bench)
    info = json.load(open(f"{self.D.R}/ckpt{ck}/{sub}/samples_meta.json"))[str(s["i"])]
    tw, th = 256, 148
    def lab(im, txt):
        im = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA); cv2.rectangle(im, (0, 0), (tw, 18), (20, 20, 20), -1)
        cv2.putText(im, txt, (5, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA); return im
    cols = []
    bmeta = {mid: json.load(open(f"{CMP_ROOT}/{mid}/{folder}/{name}/meta.json")).get("gt_crop_box_xyxy") for mid, _ in MODELS}
    for k in ks:
        gt = cut(cv2.imread(f"{info['seq_dir'].rstrip('/')}/images/{s['start'] + k:06d}.png"), None)
        ours = self.frame(f"{self.D.R}/ckpt{ck}/{sub}/sample_{s['i']:02d}.mp4", k, None)
        col = [lab(gt, f"GT f{k}"), lab(ours, f"Ours {s['lp'][0, k]:.3f}")]
        for j, (mid, ml) in enumerate(MODELS):   # every baseline at the SAME frame k, one row per model
            col.append(lab(self.frame(f"{CMP_ROOT}/{mid}/{folder}/{name}/video.mp4", k, bmeta[mid]), f"{ml} {s['lp'][j + 1, k]:.3f}"))
        cols.append(np.concatenate(col, 0))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, np.concatenate(cols, 1), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out


Cherry.spread = _spread
Cherry.strip = _strip


def _dyn(self, ck, bench):
    """per-sample GT scene dynamics (cached in cache/cherry/dyn_<bench>.json):
    img = mean |gray diff| between GT frames 8 apart (common crop, 0-255), yaw = sum |dyaw| (deg), path = metres walked."""
    p = f"{ROOT}/dyn_{bench}.json"
    names = {x["name"] for x in self.load(ck, bench)}
    if bench in self.__dict__.setdefault("_dyn", {}) and names <= self._dyn[bench].keys():
        return self._dyn[bench]
    d = json.load(open(p)) if os.path.exists(p) else {}
    _, folder, sub, *_ = sub_for(ck, bench)
    meta = json.load(open(f"{self.D.R}/ckpt{ck}/{sub}/samples_meta.json"))
    changed = False
    for s in self.load(ck, bench):
        if s["name"] in d:
            continue
        info = meta[str(s["i"])]; sd = info["seq_dir"].rstrip("/"); st0 = s["start"]
        g = [cv2.cvtColor(cut(cv2.imread(f"{sd}/images/{st0 + k:06d}.png"), None), cv2.COLOR_BGR2GRAY).astype(np.float32) for k in range(0, NF, 8)]
        img = float(np.mean([np.abs(a - b).mean() for a, b in zip(g[:-1], g[1:])]))
        P = np.load(f"{sd}/poses.npy")[st0:st0 + NF]
        yaw = float(np.abs((np.diff(P[:, 5]) + 180) % 360 - 180).sum()); path = float(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1).sum())
        d[s["name"]] = {"img": round(img, 2), "yaw": round(yaw, 1), "path": round(path, 2)}; changed = True
    if changed:
        json.dump(d, open(p, "w"))
    self._dyn[bench] = d
    return d


Cherry.dyn = _dyn
_spread_orig = Cherry.spread


def _spread_dyn(self, ck, bench, t, m, kbeat, n, gmin, need_psnr, kmin, dmin=0.0):
    out = json.loads(_spread_orig(self, ck, bench, t, m, kbeat, n, gmin, need_psnr, kmin))
    dyn = {b: self.dyn(ck, b) for b in (BENCHES if bench == "all" else [bench])}
    rows = []
    for r in out["rows"]:
        r.update(dyn[r["bench"]].get(r["name"], {"img": 0, "yaw": 0, "path": 0}))
        if r["img"] >= dmin:
            rows.append(r)
    out["rows"] = rows; out["params"]["dmin"] = dmin
    allv = sorted(v["img"] for b in dyn.values() for v in b.values())
    out["dyn_pct"] = {q: allv[int(q / 100 * (len(allv) - 1))] for q in (25, 50, 75, 90)} if allv else {}
    return json.dumps(out, separators=(",", ":")).encode()


Cherry.spread = _spread_dyn


# ---- continuity-aware selection (rev 3) -------------------------------------------------------------------------
def _gtpose(self, ck, bench):
    """name -> (yaw[77] deg unwrapped, xy[77,2], cut flag) from the GT poses of the window."""
    c = self.__dict__.setdefault("_gtp", {})
    ck_key = (ck, bench, len(self.load(ck, bench)))   # re-built when more samples land in the cache
    if ck_key in c:
        return c[ck_key]
    _, folder, sub, *_ = sub_for(ck, bench)
    meta = json.load(open(f"{self.D.R}/ckpt{ck}/{sub}/samples_meta.json")); out = {}
    for s in self.load(ck, bench):
        sd = meta[str(s["i"])]["seq_dir"].rstrip("/").replace("/lookout_val_cv/", "/lookout_val_cv_lh/")   # LookOut: L/R-corrected GT
        P = np.load(sd + "/poses.npy")[s["start"]:s["start"] + NF]
        dy = (np.diff(P[:, 5]) + 180) % 360 - 180; yaw = np.concatenate([[0], np.cumsum(dy)])
        step = np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1)
        out[s["name"]] = (yaw, P[:, :2] - P[0, :2], bool((step > 1.0).any() or (np.abs(dy) > 30).any()))
    c[ck_key] = out
    return out


def chain_pick(Q, n, yaw, gmin, ymax):
    """n frames from Q (sorted): consecutive gap >= g and |yaw step| <= ymax, maximising g, then the total yaw range.
    -> (frames, min gap) or (None, 0)."""
    Q = list(Q)
    def feasible(g):
        # best[i][c] = (yaw span so far, prev) for paths of c+1 frames ending at Q[i] (max span for ties)
        L = len(Q); best = [[None] * n for _ in range(L)]
        for i in range(L):
            best[i][0] = (0.0, None, yaw[Q[i]], yaw[Q[i]])
            for j in range(i):
                if Q[i] - Q[j] < g or abs(yaw[Q[i]] - yaw[Q[j]]) > ymax:
                    continue
                for c in range(1, n):
                    pj = best[j][c - 1]
                    if pj is None:
                        continue
                    lo, hi = min(pj[2], yaw[Q[i]]), max(pj[3], yaw[Q[i]])
                    if best[i][c] is None or hi - lo > best[i][c][0]:
                        best[i][c] = (hi - lo, j, lo, hi)
        ends = [(best[i][n - 1][0], i) for i in range(L) if best[i][n - 1] is not None]
        if not ends:
            return None
        i = max(ends)[1]; path = []; c = n - 1
        while i is not None:
            path.append(Q[i]); i = best[i][c][1]; c -= 1
        return path[::-1]
    if len(Q) < n:
        return None, 0
    lo, hi, sel = max(gmin, 1), Q[-1] - Q[0], None
    while lo <= hi:
        g = (lo + hi) // 2; p = feasible(g)
        if p:
            sel, lo = p, g + 1
        else:
            hi = g - 1
    if not sel:
        return None, 0
    return sel, int(min(np.diff(sel)))


def _spread3(self, ck, bench, t, m, kbeat, n, gmin, need_psnr, kmin, dmin=0.0, ymax=35.0, allow_cut=False, bmin=0.0):
    benches = BENCHES if bench == "all" else [bench]
    rows, nsamp, ncut = [], 0, 0
    for b in benches:
        gtp = self._gtpose(ck, b); dyn = self.dyn(ck, b)
        for s in self.load(ck, b):
            nsamp += 1
            yaw, xy, cut_ = gtp[s["name"]]
            if cut_ and not allow_cut:
                ncut += 1; continue
            lp, ps = s["lp"], s["ps"]; o, base = lp[0], lp[1:]
            nbeat = np.sum((base - o[None]) >= m, axis=0)
            ok = np.isfinite(o) & (o <= t) & (nbeat >= kbeat)
            ok &= np.nanmin(np.where(np.isnan(base), np.inf, base), axis=0) >= bmin   # even the closest baseline looks clearly off
            if need_psnr:
                ok &= ps[0] > np.nanmax(ps[1:], axis=0)
            ok[:kmin] = False
            Q = [int(k) for k in np.nonzero(ok)[0]]
            sel, g = chain_pick(Q, n, yaw, gmin, ymax)
            if sel is None:
                continue
            dv = dyn.get(s["name"], {"img": 0, "yaw": 0, "path": 0})
            if dv["img"] < dmin:
                continue
            med = np.nanmedian(base, axis=0); bestv = np.nanmin(base, axis=0)
            fr = [{"k": k, "gt_frame": s["start"] + k, "ours_lp": round(float(o[k]), 4), "med_lp": round(float(med[k]), 4),
                   "best_lp": round(float(bestv[k]), 4), "nbeat": int(nbeat[k]), "ours_ps": round(float(ps[0][k]), 2),
                   "dyaw": round(float(yaw[k] - yaw[sel[0]]), 1), "dist": round(float(np.linalg.norm(xy[k] - xy[sel[0]])), 2)} for k in sel]
            steps = [round(float(yaw[b2] - yaw[a2]), 1) for a2, b2 in zip(sel[:-1], sel[1:])]
            rows.append({"bench": b, "label": LABEL[b], "i": s["i"], "name": s["name"], "frames": fr, "n_ok": len(Q), "min_gap": g,
                         "span": sel[-1] - sel[0], "yaw_steps": steps, "yaw_range": round(float(np.ptp(yaw[sel])), 1),
                         "ours_lp": round(float(np.mean([f["ours_lp"] for f in fr])), 4),
                         "margin_med": round(float(np.mean([f["med_lp"] - f["ours_lp"] for f in fr])), 4),
                         "margin_best": round(float(np.mean([f["best_lp"] - f["ours_lp"] for f in fr])), 4),
                         "sample_ours_lp": round(float(np.nanmean(o)), 4), **dv})
    allv = sorted(v["img"] for b in benches for v in self.dyn(ck, b).values())
    return json.dumps({"ck": ck, "bench": bench, "n_samples": nsamp, "n_cut": ncut,
                       "params": {"t": t, "m": m, "kbeat": kbeat, "n": n, "gmin": gmin, "psnr": need_psnr, "kmin": kmin, "dmin": dmin, "ymax": ymax, "bmin": bmin},
                       "dyn_pct": {q: allv[int(q / 100 * (len(allv) - 1))] for q in (25, 50, 75, 90)} if allv else {},
                       "rows": rows}, separators=(",", ":")).encode()


Cherry._gtpose = _gtpose
Cherry.spread = _spread3
_strip_orig = Cherry.strip


def _strip3(self, ck, bench, name, ks):
    """as before, plus the GT row label carries yaw / distance relative to the first selected frame."""
    if bench not in BENCHES or not all(c.isalnum() or c in "_.-" for c in name) or not ks or len(ks) > 12 or not all(0 <= k < NF for k in ks):
        return None
    out = f"{ROOT}/strips_v3/ck{ck}/{bench}/{name}_{'-'.join(map(str, ks))}.jpg"
    if os.path.exists(out):
        return out
    p = _strip_orig(self, ck, bench, name, ks)
    if not p:
        return None
    yaw, xy, _ = self._gtpose(ck, bench)[name]
    im = cv2.imread(p); tw = 256
    for c, k in enumerate(ks):
        cv2.rectangle(im, (c * tw, 0), (c * tw + tw, 18), (20, 20, 20), -1)
        txt = f"GT f{k}  yaw {yaw[k] - yaw[ks[0]]:+.0f}deg  {np.linalg.norm(xy[k] - xy[ks[0]]):.1f}m"
        cv2.putText(im, txt, (c * tw + 5, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(out), exist_ok=True); cv2.imwrite(out, im, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out


Cherry.strip = _strip3


def _strip4(self, ck, bench, name, ks):
    """rows = selected frames (top to bottom), columns = GT | ours | 7 baselines at that frame.
    GT tile label: frame + yaw / distance relative to the first selected frame; other tiles: model + LPIPS vs GT."""
    if bench not in BENCHES or not all(c.isalnum() or c in "_.-" for c in name) or not ks or len(ks) > 12 or not all(0 <= k < NF for k in ks):
        return None
    out = f"{ROOT}/strips_v4/ck{ck}/{bench}/{name}_{'-'.join(map(str, ks))}.jpg"
    if os.path.exists(out):
        return out
    s = next((x for x in self.load(ck, bench) if x["name"] == name), None)
    if s is None:
        return None
    _, folder, sub, *_ = sub_for(ck, bench)
    info = json.load(open(f"{self.D.R}/ckpt{ck}/{sub}/samples_meta.json"))[str(s["i"])]
    yaw, xy, _ = self._gtpose(ck, bench)[name]
    bmeta = {mid: json.load(open(f"{CMP_ROOT}/{mid}/{folder}/{name}/meta.json")).get("gt_crop_box_xyxy") for mid, _ in MODELS}
    tw, th = 256, 148
    def lab(im, txt):
        im = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA); cv2.rectangle(im, (0, 0), (tw, 18), (20, 20, 20), -1)
        cv2.putText(im, txt, (5, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA); return im
    rows = []
    for k in ks:
        gt = cut(cv2.imread(f"{info['seq_dir'].rstrip('/')}/images/{s['start'] + k:06d}.png"), None)
        row = [lab(gt, f"GT f{k} yaw {yaw[k] - yaw[ks[0]]:+.0f}deg {np.linalg.norm(xy[k] - xy[ks[0]]):.1f}m"),
               lab(self.frame(f"{self.D.R}/ckpt{ck}/{sub}/sample_{s['i']:02d}.mp4", k, None), f"Ours {s['lp'][0, k]:.3f}")]
        for j, (mid, ml) in enumerate(MODELS):
            row.append(lab(self.frame(f"{CMP_ROOT}/{mid}/{folder}/{name}/video.mp4", k, bmeta[mid]), f"{ml} {s['lp'][j + 1, k]:.3f}"))
        rows.append(np.concatenate(row, 1))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out


Cherry.strip = _strip4


# ---- route (BEV) + export (rev 5) --------------------------------------------------------------------------------
EXPORT = "/root/project/egowam/results_syn/goalprog_mode_real6_nomr_hg0_grad0.5/cherry_export"


def _ego_actions(p):
    p = np.asarray(p, float); dxw, dyw = p[1:, 0] - p[:-1, 0], p[1:, 1] - p[:-1, 1]; yaw = np.deg2rad(p[:-1, 5])
    return np.stack([dxw * np.cos(yaw) + dyw * np.sin(yaw), -dxw * np.sin(yaw) + dyw * np.cos(yaw), (p[1:, 5] - p[:-1, 5] + 180) % 360 - 180], -1)


def _integrate(a, x0, y0, yaw0):
    out, x, y, yaw = [(x0, y0, yaw0)], x0, y0, float(yaw0)
    for dx, dy, dyaw in a:
        r = math.radians(yaw); x += dx * math.cos(r) - dy * math.sin(r); y += dx * math.sin(r) + dy * math.cos(r); yaw += dyaw
        out.append((x, y, yaw))
    return np.array(out)


def _tracks(self, ck, bench, name):
    """GT, ours (predicted camera -> ego actions integrated from the GT start, as in cam77) and ours-video (DA3 on the
    generated RGB, Sim3-fitted to GT) in the start-frame ego frame: (right, forward) metres, heading up."""
    s = next(x for x in self.load(ck, bench) if x["name"] == name)
    _, folder, sub, *_ = sub_for(ck, bench); D = f"{self.D.R}/ckpt{ck}/{sub}"
    info = json.load(open(f"{D}/samples_meta.json"))[str(s["i"])]
    sd = info["seq_dir"].rstrip("/").replace("/lookout_val_cv/", "/lookout_val_cv_lh/")   # LookOut: L/R-corrected GT
    P = np.load(f"{sd}/poses.npy"); st0 = s["start"]; g = P[st0:st0 + NF]
    goal = info.get("goal_frame"); gpt = P[int(goal), :2] if goal is not None and int(goal) < len(P) else None
    x0, y0, yaw0 = g[0, 0], g[0, 1], g[0, 5]
    def ego(xy):
        d = np.asarray(xy, float)[:, :2] - [x0, y0]; r = math.radians(yaw0)
        return np.stack([-d[:, 0] * math.sin(r) + d[:, 1] * math.cos(r), d[:, 0] * math.cos(r) + d[:, 1] * math.sin(r)], -1)
    out = {"gt": ego(g), "goal": ego(gpt[None])[0] if gpt is not None else None}
    pp = f"{D}/sample_{s['i']:02d}_pred_poses.npy"
    if os.path.exists(pp):
        out["ours"] = ego(_integrate(_ego_actions(np.load(pp)[:NF]), x0, y0, yaw0))
    # DA3 camera of the generated video, Sim3-fitted to GT (LookOut: the _lh re-fit to the corrected GT)
    dz = f"{D}_lh/viz_da3_fix/sample_{s['i']:02d}_da3_poses.npy" if "lookout" in bench and not D.endswith("_lh") else f"{D}/viz_da3_fix/sample_{s['i']:02d}_da3_poses.npy"
    if os.path.exists(dz):
        out["video"] = ego(_da3_xy(np.load(dz)[:NF]))
    return out


def _da3_xy(c2w):
    """DA3 c2w (opencv cam, fitted into the GT world built by eval_suite: carla_pose_to_mat @ C_CV2CARLA) -> world xy."""
    return c2w[:, :2, 3]


def _route(self, ck, bench, name, ks):
    if bench not in BENCHES or not all(c.isalnum() or c in "_.-" for c in name) or not ks or len(ks) > 12 or not all(0 <= k < NF for k in ks):
        return None
    out = f"{ROOT}/routes/ck{ck}/{bench}/{name}_{'-'.join(map(str, ks))}.png"
    if os.path.exists(out):
        return out
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    T = self._tracks(ck, bench, name)
    fig, ax = plt.subplots(figsize=(3.6, 3.6), dpi=110)
    ax.plot(T["gt"][:, 0], T["gt"][:, 1], "-", color="#222", lw=2.2, label="GT")
    if "ours" in T:
        ax.plot(T["ours"][:, 0], T["ours"][:, 1], "-", color="#e8590c", lw=1.8, label="ours: predicted action")
    if "video" in T:
        ax.plot(T["video"][:, 0], T["video"][:, 1], "--", color="#2f6fd6", lw=1.3, label="ours: camera in video (DA3)")
    ax.plot(0, 0, "o", color="#2b8a3e", ms=7, label="start")
    if T["goal"] is not None:
        ax.plot(*T["goal"], "*", color="#c92a2a", ms=13, label="goal")
    for n, k in enumerate(ks):
        for key, col in (("gt", "#222"), ("ours", "#e8590c")):
            if key in T:
                ax.plot(*T[key][k], "o", ms=5, mfc="white", mec=col, mew=1.5)
        ax.annotate(str(n + 1), T["gt"][k], textcoords="offset points", xytext=(5, 4), fontsize=8, color="#222")
    ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=.3)
    ax.set_xlabel("right (m)", fontsize=8); ax.set_ylabel("forward (m)", fontsize=8); ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc="best", framealpha=.85)
    fig.tight_layout(); os.makedirs(os.path.dirname(out), exist_ok=True); fig.savefig(out); plt.close(fig)
    return out


def _export(self, ck, bench, name, ks):
    """cherry_export/<bench>/<sample>/<model>/frames/fKK.png (native resolution) for the selected frames,
    + <sample>/route.png + <sample>/meta.json. model = gt | ours_ck<ck> | baseline id."""
    if bench not in BENCHES or not all(c.isalnum() or c in "_.-" for c in name) or not ks or not all(0 <= k < NF for k in ks):
        return None
    s = next((x for x in self.load(ck, bench) if x["name"] == name), None)
    if s is None:
        return None
    _, folder, sub, *_ = sub_for(ck, bench); D = f"{self.D.R}/ckpt{ck}/{sub}"
    info = json.load(open(f"{D}/samples_meta.json"))[str(s["i"])]
    root = f"{EXPORT}{'' if ck == 7500 else f'_ck{ck}'}/{bench}/{name}"   # non-7500 ckpts export to their own tree
    def raw(p, k):
        cap = cv2.VideoCapture(p); cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = cap.read(); return f if ok else None
    srcs = [("gt", None), (f"ours_ck{ck}", f"{D}/sample_{s['i']:02d}.mp4")] + [(mid, f"{CMP_ROOT}/{mid}/{folder}/{name}/video.mp4") for mid, _ in MODELS]
    for model, vid in srcs:
        od = f"{root}/{model}/frames"; os.makedirs(od, exist_ok=True)
        for k in ks:
            im = cv2.imread(f"{info['seq_dir'].rstrip('/')}/images/{s['start'] + k:06d}.png") if vid is None else raw(vid, k)
            if im is not None:
                cv2.imwrite(f"{od}/f{k:02d}.png", im)
    r = self.route(ck, bench, name, ks)
    if r:
        import shutil; shutil.copy(r, f"{root}/route.png")
    yaw, xy, _ = self._gtpose(ck, bench)[name]
    json.dump({"bench": bench, "sample": name, "sample_index": s["i"], "ckpt": ck, "frames": ks,
               "gt_frames": [s["start"] + k for k in ks], "seq_dir": info["seq_dir"],
               "yaw_deg_vs_first": [round(float(yaw[k] - yaw[ks[0]]), 1) for k in ks],
               "dist_m_vs_first": [round(float(np.linalg.norm(xy[k] - xy[ks[0]])), 2) for k in ks],
               "lpips_vs_gt": {m: [round(float(s["lp"][j, k]), 4) for k in ks] for j, m in enumerate([f"ours_ck{ck}"] + [x for x, _ in MODELS])},
               "psnr_vs_gt": {m: [round(float(s["ps"][j, k]), 2) for k in ks] for j, m in enumerate([f"ours_ck{ck}"] + [x for x, _ in MODELS])},
               "note": "frames = native resolution of each source (GT 512x512 png; baselines at their own size/crop); lpips/psnr were measured on the common GT rows 108..404"},
              open(f"{root}/meta.json", "w"), indent=1)
    return root


Cherry._tracks = _tracks
Cherry.route = _route
Cherry.export = _export


# ---- concept route for figures (rev 6) ---------------------------------------------------------------------------
def _describe(yaw, xy, ks):
    """short Korean description of what the camera does between consecutive selected frames."""
    parts = []
    for n, (a, b) in enumerate(zip(ks[:-1], ks[1:])):
        dy = float(yaw[b] - yaw[a]); d = float(np.linalg.norm(xy[b] - xy[a]))
        if abs(dy) >= 10:
            txt = f"{'오른쪽' if dy > 0 else '왼쪽'}으로 {abs(dy):.0f}° 회전" + (f"하며 {d:.1f}m 이동" if d >= 0.5 else "")
        else:
            txt = f"직진 {d:.1f}m" if d >= 0.3 else "거의 제자리"
        parts.append((n + 1, n + 2, txt))
    merged = []   # merge consecutive identical "직진"/"제자리" steps
    for a, b, t in parts:
        if merged and t.startswith("직진") and merged[-1][2].startswith("직진"):
            d = float(merged[-1][2].split()[1][:-1]) + float(t.split()[1][:-1]); merged[-1] = (merged[-1][0], b, f"직진 {d:.1f}m")
        elif merged and t == merged[-1][2] == "거의 제자리":
            merged[-1] = (merged[-1][0], b, t)
        else:
            merged.append((a, b, t))
    tot = float(yaw[ks[-1]] - yaw[ks[0]])
    return {"steps": [f"{a}→{b} {t}" for a, b, t in merged], "total_yaw": round(tot, 1),
            "total_dist": round(float(np.linalg.norm(xy[ks[-1]] - xy[ks[0]])), 2),
            "text": " · ".join(f"{a}→{b} {t}" for a, b, t in merged)}


def _route_concept(self, ck, bench, name, ks):
    if bench not in BENCHES or not all(c.isalnum() or c in "_.-" for c in name) or not ks or len(ks) > 12 or not all(0 <= k < NF for k in ks):
        return None
    out = f"{ROOT}/routes_concept/ck{ck}/{bench}/{name}_{'-'.join(map(str, ks))}.png"
    if os.path.exists(out):
        return out
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge, FancyArrowPatch
    T = self._tracks(ck, bench, name); g = T["gt"]
    yaw, _, _ = self._gtpose(ck, bench)[name]; yaw = yaw - yaw[0]       # relative heading, + = right turn
    gs = np.copy(g)
    for i in range(len(g)):                                             # light smoothing for a schematic line
        a, b = max(0, i - 3), min(len(g), i + 4); gs[i] = g[a:b].mean(0)
    span = max(np.ptp(gs[:, 0]), np.ptp(gs[:, 1]), 1.0); R = 0.16 * span
    cmap = plt.get_cmap("viridis"); cols = [cmap(0.1 + 0.8 * n / max(len(ks) - 1, 1)) for n in range(len(ks))]
    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=150)
    ax.plot(gs[:, 0], gs[:, 1], "-", color="#8a8f98", lw=2.4, solid_capstyle="round", zorder=1)
    ax.add_patch(FancyArrowPatch(gs[-6], gs[-1], arrowstyle="-|>", mutation_scale=14, color="#8a8f98", lw=2.4, zorder=1))
    ax.plot(*gs[0], "o", color="#444", ms=6, zorder=3)
    if T["goal"] is not None:
        ax.plot(*T["goal"], marker="*", color="#d6336c", ms=15, mec="white", mew=0.8, zorder=4)
    for n, k in enumerate(ks):
        p = g[k]; h = float(yaw[k])                                     # heading: angle from +forward, + = right
        ang = 90 - h                                                    # matplotlib degrees from +x (right)
        ax.add_patch(Wedge(p, R, ang - 46.5, ang + 46.5, facecolor=cols[n], alpha=0.28, edgecolor=cols[n], lw=1.2, zorder=2))
        ax.plot(*p, "o", ms=11, color=cols[n], mec="white", mew=1.2, zorder=5)
        ax.text(*p, str(n + 1), ha="center", va="center", fontsize=7, color="white", fontweight="bold", zorder=6)
    pts = np.concatenate([gs, g[ks]] + ([T["goal"][None]] if T["goal"] is not None else []))
    pad = R * 1.15; ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad); ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(pad=0.1); os.makedirs(os.path.dirname(out), exist_ok=True); fig.savefig(out, transparent=True); plt.close(fig)
    return out


_route_detail = Cherry.route
_export_v5 = Cherry.export


def _export6(self, ck, bench, name, ks):
    root = _export_v5(self, ck, bench, name, ks)
    if not root:
        return None
    import shutil
    d = _route_detail(self, ck, bench, name, ks)
    if d:
        shutil.copy(d, f"{root}/route_detail.png")
    c = _route_concept(self, ck, bench, name, ks)
    if c:
        shutil.copy(c, f"{root}/route.png")
    yaw, xy, _ = self._gtpose(ck, bench)[name]
    m = json.load(open(f"{root}/meta.json")); m["description"] = _describe(yaw, xy, ks)
    m["route_png"] = "route.png = schematic (GT path, numbered view cones at the selected frames, start dot, goal star); route_detail.png = GT vs ours action vs ours video camera"
    json.dump(m, open(f"{root}/meta.json", "w"), indent=1, ensure_ascii=False)
    return root


Cherry.route_detail = _route_detail
Cherry.route = _route_concept
Cherry.export = _export6
_spread_v3 = Cherry.spread


def _spread6(self, ck, bench, *a, **kw):
    out = json.loads(_spread_v3(self, ck, bench, *a, **kw))
    for r in out["rows"]:
        yaw, xy, _ = self._gtpose(ck, r["bench"])[r["name"]]
        r["desc"] = _describe(yaw, xy, [f["k"] for f in r["frames"]])["text"]
    return json.dumps(out, separators=(",", ":"), ensure_ascii=False).encode()


Cherry.spread = _spread6


# ---- concept route v2 (rev 7): cleaner schematic ----------------------------------------------------------------
BLUE_ORD = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]   # ordinal blue ramp (light -> dark)


def _route_concept2(self, ck, bench, name, ks):
    if bench not in BENCHES or not all(c.isalnum() or c in "_.-" for c in name) or not ks or len(ks) > 12 or not all(0 <= k < NF for k in ks):
        return None
    out = f"{ROOT}/routes_concept2/ck{ck}/{bench}/{name}_{'-'.join(map(str, ks))}.png"
    if os.path.exists(out):
        return out
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    T = self._tracks(ck, bench, name); g = T["gt"].copy()
    yaw, _, _ = self._gtpose(ck, bench)[name]; yaw = yaw - yaw[0]
    gs = np.copy(g)
    for i in range(len(g)):
        a, b = max(0, i - 4), min(len(g), i + 5); gs[i] = g[a:b].mean(0)
    # rotate so the overall start -> end displacement points straight up
    d = gs[-1] - gs[0]; th = math.atan2(d[0], d[1]) if np.linalg.norm(d) > 0.3 else 0.0
    Rm = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    rot = lambda p: np.asarray(p) @ Rm.T
    gs, gk = rot(gs), rot(g[ks]); goal = rot(T["goal"]) if T["goal"] is not None else None
    head = np.deg2rad(yaw[ks]) - th                                   # heading angle from +up, + = clockwise (right)
    span = max(np.ptp(gs[:, 1]), np.ptp(gs[:, 0]), 1.0)
    nb = [min([np.linalg.norm(gk[i] - gk[j]) for j in range(len(ks)) if j != i] or [span]) for i in range(len(ks))]
    cols = [BLUE_ORD[min(int(round(n * (len(BLUE_ORD) - 1) / max(len(ks) - 1, 1))), len(BLUE_ORD) - 1)] for n in range(len(ks))]
    fig, ax = plt.subplots(figsize=(2.8, 3.4), dpi=200)
    ax.plot(gs[:, 0], gs[:, 1], "-", color="#c3c8d0", lw=2.0, solid_capstyle="round", zorder=1)
    v = gs[-1] - gs[-4]; v = v / (np.linalg.norm(v) + 1e-9); s = 0.035 * span; p = gs[-1]; n_ = np.array([-v[1], v[0]])
    ax.add_patch(Polygon([p + v * s, p - v * s * .6 + n_ * s * .7, p - v * s * .6 - n_ * s * .7], closed=True, color="#c3c8d0", zorder=1))
    ax.plot(*gs[0], "o", ms=6, mfc="white", mec="#8d949e", mew=1.4, zorder=3)
    if goal is not None:
        ax.plot(*goal, marker="*", ms=11, color="#3b3f46", mec="white", mew=0.6, zorder=4)
    for n, k in enumerate(ks):
        L = float(np.clip(0.55 * nb[n], 0.07 * span, 0.17 * span)); h = head[n]; p = gk[n]
        dirs = [h - math.radians(46.5), h + math.radians(46.5)]
        tri = [p] + [p + L * np.array([math.sin(a), math.cos(a)]) for a in dirs]
        ax.add_patch(Polygon(tri, closed=True, facecolor=cols[n], alpha=0.16, edgecolor="none", zorder=2))
        for a in dirs:
            ax.plot([p[0], p[0] + L * math.sin(a)], [p[1], p[1] + L * math.cos(a)], "-", color=cols[n], lw=1.1, alpha=0.9, zorder=2)
        ax.plot(*p, "o", ms=9.5, color=cols[n], mec="white", mew=1.3, zorder=5)
        ax.text(*p, str(n + 1), ha="center", va="center", fontsize=6.2, color="white", fontweight="bold", zorder=6)
    pts = np.concatenate([gs, gk] + ([goal[None]] if goal is not None else []))
    pad = 0.18 * span; cx = (pts[:, 0].min() + pts[:, 0].max()) / 2
    half = max((pts[:, 0].max() - pts[:, 0].min()) / 2 + pad, (pts[:, 1].max() - pts[:, 1].min() + 2 * pad) * 2.8 / 3.4 / 2)
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    ax.set_aspect("equal", adjustable="datalim"); ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1); os.makedirs(os.path.dirname(out), exist_ok=True); fig.savefig(out, transparent=True); plt.close(fig)
    return out


Cherry.route = _route_concept2
_route_concept = _route_concept2      # export6 copies this one as route.png


# ---- concept route v3 (rev 8): turtle schematic, not metric --------------------------------------------------------
def _route_concept3(self, ck, bench, name, ks):
    """Schematic for figures: markers 1..n placed at a constant step along the camera heading (a turn bends the path),
    a short step where the camera mostly rotates in place; equal-size view cones (HFOV ~93 deg); big turns labelled."""
    if bench not in BENCHES or not all(c.isalnum() or c in "_.-" for c in name) or not ks or len(ks) > 12 or not all(0 <= k < NF for k in ks):
        return None
    out = f"{ROOT}/routes_concept3b/ck{ck}/{bench}/{name}_{'-'.join(map(str, ks))}.png"
    if os.path.exists(out):
        return out
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    yaw, xy, _ = self._gtpose(ck, bench)[name]
    h = np.deg2rad(np.array([yaw[k] - yaw[ks[0]] for k in ks]))           # + = right turn
    P = [np.zeros(2)]
    for n in range(len(ks) - 1):
        dist = float(np.linalg.norm(xy[ks[n + 1]] - xy[ks[n]])); step = 1.0 if dist >= 0.3 else 0.45
        a = (h[n] + h[n + 1]) / 2; P.append(P[-1] + step * np.array([math.sin(a), math.cos(a)]))
    P = np.array(P)
    cols = [BLUE_ORD[min(int(round(n * (len(BLUE_ORD) - 1) / max(len(ks) - 1, 1))), len(BLUE_ORD) - 1)] for n in range(len(ks))]
    L = 0.46
    fig, ax = plt.subplots(figsize=(2.6, 3.2), dpi=220)
    # smooth path: quadratic beziers through the mid-points, with a lead-in and an arrow tip
    lead = P[0] - 0.45 * np.array([math.sin(h[0]), math.cos(h[0])]); tail = P[-1] + 0.55 * np.array([math.sin(h[-1]), math.cos(h[-1])])
    Q = np.vstack([lead, P, tail]); verts, codes = [Q[0]], [Path.MOVETO]
    for i in range(1, len(Q) - 1):
        m = (Q[i] + Q[i + 1]) / 2; verts += [Q[i], m]; codes += [Path.CURVE3, Path.CURVE3]
    verts.append(Q[-1]); codes.append(Path.LINETO)
    ax.add_patch(PathPatch(Path(verts, codes), facecolor="none", edgecolor="#c3c8d0", lw=2.2, capstyle="round", zorder=1))
    v = np.array([math.sin(h[-1]), math.cos(h[-1])]); nrm = np.array([-v[1], v[0]]); s = 0.12
    ax.add_patch(Polygon([tail + v * s, tail - v * s * .5 + nrm * s * .75, tail - v * s * .5 - nrm * s * .75], closed=True, color="#c3c8d0", zorder=1))
    for n in range(len(ks)):
        p = P[n]; dirs = [h[n] - math.radians(46.5), h[n] + math.radians(46.5)]
        tri = [p] + [p + L * np.array([math.sin(a), math.cos(a)]) for a in dirs]
        ax.add_patch(Polygon(tri, closed=True, facecolor=cols[n], alpha=0.12, edgecolor="none", zorder=2))
        for a in dirs:
            ax.plot([p[0], p[0] + L * math.sin(a)], [p[1], p[1] + L * math.cos(a)], "-", color=cols[n], lw=1.0, alpha=0.85, zorder=2, solid_capstyle="round")
        ax.plot(*p, "o", ms=10, color=cols[n], mec="white", mew=1.4, zorder=5)
        ax.text(*p, str(n + 1), ha="center", va="center", fontsize=6.5, color="white", fontweight="bold", zorder=6)
    for n in range(len(ks) - 1):                                          # label the big turns, beside the segment
        dy = math.degrees(h[n + 1] - h[n])
        if abs(dy) >= 12:
            mid = (P[n] + P[n + 1]) / 2; a = (h[n] + h[n + 1]) / 2; side = -1 if dy > 0 else 1   # text on the outer side
            off = side * 0.38 * np.array([math.cos(a), -math.sin(a)])
            ax.text(*(mid + off), f"{abs(dy):.0f}° {'right' if dy > 0 else 'left'}", ha="center", va="center", fontsize=6, color="#5c6577")
    pts = np.vstack([Q, P + L, P - L * 0.2])
    ax.set_xlim(pts[:, 0].min() - 0.5, pts[:, 0].max() + 0.5); ax.set_ylim(Q[:, 1].min() - 0.3, max(Q[:, 1].max(), (P + L)[:, 1].max()) + 0.3)
    ax.set_aspect("equal", adjustable="datalim"); ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1); os.makedirs(os.path.dirname(out), exist_ok=True); fig.savefig(out, transparent=True); plt.close(fig)
    return out


Cherry.route = _route_concept3
_route_concept = _route_concept3


# ---- concept route v4 (rev 9): "fancy" schematic --------------------------------------------------------------------
def _route_concept4(self, ck, bench, name, ks):
    if bench not in BENCHES or not all(c.isalnum() or c in "_.-" for c in name) or not ks or len(ks) > 12 or not all(0 <= k < NF for k in ks):
        return None
    out = f"{ROOT}/routes_concept4/ck{ck}/{bench}/{name}_{'-'.join(map(str, ks))}.png"
    if os.path.exists(out):
        return out
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Circle
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgb
    import matplotlib.patheffects as pe
    yaw, xy, _ = self._gtpose(ck, bench)[name]
    h = np.deg2rad(np.array([yaw[k] - yaw[ks[0]] for k in ks]))
    P = [np.zeros(2)]
    for n in range(len(ks) - 1):
        dist = float(np.linalg.norm(xy[ks[n + 1]] - xy[ks[n]])); step = 1.0 if dist >= 0.3 else 0.5
        a = (h[n] + h[n + 1]) / 2; P.append(P[-1] + step * np.array([math.sin(a), math.cos(a)]))
    P = np.array(P); N = len(ks)
    ramp = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    def col(t):                                                         # t in [0,1] along the ramp
        x = t * (len(ramp) - 1); i = min(int(x), len(ramp) - 2); f = x - i
        return tuple((1 - f) * np.array(to_rgb(ramp[i])) + f * np.array(to_rgb(ramp[i + 1])))
    cols = [col(n / max(N - 1, 1)) for n in range(N)]
    # smooth path (quadratic beziers through mid-points), sampled densely, with a lead-in and a tail
    u0 = np.array([math.sin(h[0]), math.cos(h[0])]); u1 = np.array([math.sin(h[-1]), math.cos(h[-1])])
    Q = np.vstack([P[0] - 0.5 * u0, P, P[-1] + 0.6 * u1])
    pts = [Q[0]]
    for i in range(1, len(Q) - 1):
        a0 = pts[-1]; c = Q[i]; b = (Q[i] + Q[i + 1]) / 2 if i < len(Q) - 2 else Q[i + 1]
        for t in np.linspace(0, 1, 30)[1:]:
            pts.append((1 - t) ** 2 * a0 + 2 * (1 - t) * t * c + t ** 2 * b)
    pts = np.array(pts); tt = np.linspace(0, 1, len(pts))
    fig, ax = plt.subplots(figsize=(2.7, 3.3), dpi=240)
    segs = np.stack([pts[:-1], pts[1:]], 1)
    ax.add_collection(LineCollection(segs + np.array([0.03, -0.04]), colors=[(0, 0, 0, 0.07)], linewidths=7, capstyle="round", zorder=1))  # soft shadow
    ax.add_collection(LineCollection(segs, colors="white", linewidths=6.2, capstyle="round", zorder=2))                                   # halo
    ax.add_collection(LineCollection(segs, colors=[col(0.15 + 0.85 * t) for t in tt[:-1]], linewidths=3.2, capstyle="round", zorder=3))
    v = pts[-1] - pts[-6]; v /= np.linalg.norm(v) + 1e-9; nv = np.array([-v[1], v[0]]); s = 0.16; tip = pts[-1] + v * 0.06
    ax.add_patch(Polygon([tip + v * s, tip - v * s * .45 + nv * s * .8, tip - v * s * .45 - nv * s * .8], closed=True, color=col(1.0),
                         ec="white", lw=1.0, zorder=4, joinstyle="round"))
    # view cones: a light beam fading with distance (stacked slices)
    L, half, K = 0.62, math.radians(46.5), 14
    for n in range(N):
        p = P[n]; c = cols[n]
        for j in range(K):
            r0, r1 = L * j / K, L * (j + 1) / K
            a0, a1 = h[n] - half, h[n] + half
            arc = lambda r: [p + r * np.array([math.sin(a), math.cos(a)]) for a in np.linspace(a0, a1, 12)]
            poly = arc(r1) + arc(r0)[::-1]
            ax.add_patch(Polygon(poly, closed=True, facecolor=c, edgecolor="none", alpha=0.30 * (1 - j / K) ** 1.6, zorder=2.5))
        for a in (h[n] - half, h[n] + half):
            e = p + 0.78 * L * np.array([math.sin(a), math.cos(a)])
            ax.add_collection(LineCollection([[p, e]], colors=[(*c, 0.55)], linewidths=0.8, zorder=2.6))
    # markers with a soft drop shadow and white ring
    R = 0.13
    for n in range(N):
        p = P[n]
        ax.add_patch(Circle(p + np.array([0.02, -0.03]), R * 1.08, color=(0, 0, 0, 0.18), zorder=5))
        ax.add_patch(Circle(p, R, facecolor=cols[n], edgecolor="white", lw=1.4, zorder=6))
        ax.text(*p, str(n + 1), ha="center", va="center", fontsize=6.6, color="white", fontweight="bold", zorder=7)
    # turn badges
    for n in range(N - 1):
        dy = math.degrees(h[n + 1] - h[n])
        if abs(dy) >= 12:
            mid = (P[n] + P[n + 1]) / 2; a = (h[n] + h[n + 1]) / 2; side = 1 if dy < 0 else -1
            pos = mid + side * 0.52 * np.array([math.cos(a), -math.sin(a)])
            ax.text(*pos, f"{'↺' if dy < 0 else '↻'} {abs(dy):.0f}°", ha="center", va="center", fontsize=6.2, color="#33415c", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.6", fc="white", ec="#d6dbe3", lw=0.7))
    # start badge
    ax.add_patch(Circle(Q[0], 0.06, facecolor="white", edgecolor=col(0.1), lw=1.4, zorder=6))
    allp = np.vstack([pts, P + L, P - L * 0.3])
    ax.set_xlim(allp[:, 0].min() - 0.55, allp[:, 0].max() + 0.55); ax.set_ylim(pts[:, 1].min() - 0.25, max(pts[:, 1].max(), (P + L)[:, 1].max()) + 0.3)
    ax.set_aspect("equal", adjustable="datalim"); ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1); os.makedirs(os.path.dirname(out), exist_ok=True); fig.savefig(out, transparent=True); plt.close(fig)
    return out


Cherry.route = _route_concept4
_route_concept = _route_concept4


# ---- concept route v5 (rev 10): fuller look, no angle badges --------------------------------------------------------
def _route_concept5(self, ck, bench, name, ks):
    if bench not in BENCHES or not all(c.isalnum() or c in "_.-" for c in name) or not ks or len(ks) > 12 or not all(0 <= k < NF for k in ks):
        return None
    out = f"{ROOT}/routes_concept5b/ck{ck}/{bench}/{name}_{'-'.join(map(str, ks))}.png"
    if os.path.exists(out):
        return out
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Circle, FancyBboxPatch
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgb
    yaw, xy, _ = self._gtpose(ck, bench)[name]
    h = np.deg2rad(np.array([yaw[k] - yaw[ks[0]] for k in ks]))
    P = [np.zeros(2)]
    for n in range(len(ks) - 1):
        dist = float(np.linalg.norm(xy[ks[n + 1]] - xy[ks[n]])); step = 1.0 if dist >= 0.3 else 0.55
        a = (h[n] + h[n + 1]) / 2; P.append(P[-1] + step * np.array([math.sin(a), math.cos(a)]))
    P = np.array(P); N = len(ks)
    ramp = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    def col(t):
        x = min(max(t, 0), 1) * (len(ramp) - 1); i = min(int(x), len(ramp) - 2); f = x - i
        return tuple((1 - f) * np.array(to_rgb(ramp[i])) + f * np.array(to_rgb(ramp[i + 1])))
    cols = [col(0.1 + 0.9 * n / max(N - 1, 1)) for n in range(N)]
    u0 = np.array([math.sin(h[0]), math.cos(h[0])]); u1 = np.array([math.sin(h[-1]), math.cos(h[-1])])
    Q = np.vstack([P[0] - 0.55 * u0, P, P[-1] + 0.7 * u1])
    pts = [Q[0]]
    for i in range(1, len(Q) - 1):
        a0 = pts[-1]; c = Q[i]; b = (Q[i] + Q[i + 1]) / 2 if i < len(Q) - 2 else Q[i + 1]
        for t in np.linspace(0, 1, 30)[1:]:
            pts.append((1 - t) ** 2 * a0 + 2 * (1 - t) * t * c + t ** 2 * b)
    pts = np.array(pts); tt = np.linspace(0, 1, len(pts)); segs = np.stack([pts[:-1], pts[1:]], 1)
    L, half, K, R = 0.9, math.radians(46.5), 18, 0.19
    allp = np.vstack([pts] + [P[n] + L * np.array([math.sin(a), math.cos(a)]) for n in range(N) for a in (h[n] - half, h[n], h[n] + half)])
    x0, x1 = allp[:, 0].min() - 0.35, allp[:, 0].max() + 0.35; y0, y1 = allp[:, 1].min() - 0.3, allp[:, 1].max() + 0.3
    w = x1 - x0; hh = y1 - y0; fig, ax = plt.subplots(figsize=(3.0, 3.0 * hh / w if hh / w < 1.6 else 4.6), dpi=240)
    # ground card with a dotted grid
    ax.add_patch(FancyBboxPatch((x0, y0), w, hh, boxstyle="round,pad=0,rounding_size=0.25", fc="#f4f7fb", ec="#e3e8ef", lw=0.8, zorder=0))
    gx, gy = np.meshgrid(np.arange(math.ceil(x0 / .35) * .35, x1, .35), np.arange(math.ceil(y0 / .35) * .35, y1, .35))
    ax.scatter(gx.ravel(), gy.ravel(), s=0.6, color="#cfd7e3", zorder=0.5, linewidths=0)
    # view beams (behind the road), fading with distance
    for n in range(N):
        p = P[n]; c = cols[n]
        for j in range(K):
            r0, r1 = L * j / K, L * (j + 1) / K
            arc = lambda r: [p + r * np.array([math.sin(a), math.cos(a)]) for a in np.linspace(h[n] - half, h[n] + half, 16)]
            ax.add_patch(Polygon(arc(r1) + arc(r0)[::-1], closed=True, facecolor=c, edgecolor="none", alpha=0.42 * (1 - j / K) ** 1.3, zorder=1))
        for a in (h[n] - half, h[n] + half):
            ax.add_collection(LineCollection([[p, p + 0.92 * L * np.array([math.sin(a), math.cos(a)])]], colors=[(*c, 0.5)], linewidths=0.8, zorder=1.1))
    # road: wide soft band + bright core with a white halo
    ax.plot(pts[:, 0], pts[:, 1], "-", color=col(0.45), alpha=0.16, lw=16, solid_capstyle="round", solid_joinstyle="round", zorder=2)   # soft band, one path (no alpha build-up)
    ax.plot(pts[:, 0], pts[:, 1], "-", color="white", lw=7.0, solid_capstyle="round", solid_joinstyle="round", zorder=3)                # thin white edge
    ax.add_collection(LineCollection(segs, colors=[col(0.1 + 0.9 * t) for t in tt[:-1]], linewidths=5.0, capstyle="butt", zorder=4))
    v = pts[-1] - pts[-6]; v /= np.linalg.norm(v) + 1e-9; nv = np.array([-v[1], v[0]]); s = 0.2; tip = pts[-1] + v * 0.05
    ax.add_patch(Polygon([tip + v * s, tip - v * s * .45 + nv * s * .8, tip - v * s * .45 - nv * s * .8], closed=True, fc=col(1.0), ec="white", lw=1.2, zorder=5))
    ax.add_patch(Circle(Q[0], 0.085, facecolor="white", edgecolor=col(0.1), lw=1.8, zorder=5))
    for n in range(N):
        p = P[n]
        ax.add_patch(Circle(p + np.array([0.025, -0.035]), R * 1.06, color=(0, 0, 0, 0.16), zorder=6))
        ax.add_patch(Circle(p, R, facecolor=cols[n], edgecolor="white", lw=1.8, zorder=7))
        ax.text(*p, str(n + 1), ha="center", va="center", fontsize=8, color="white", fontweight="bold", zorder=8)
    ax.set_xlim(x0 - 0.02, x1 + 0.02); ax.set_ylim(y0 - 0.02, y1 + 0.02); ax.set_aspect("equal"); ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1); os.makedirs(os.path.dirname(out), exist_ok=True); fig.savefig(out, transparent=True); plt.close(fig)
    return out


Cherry.route = _route_concept5
_route_concept = _route_concept5


# ---- concept route v6 (rev 11): path + camera glyphs + photo cards --------------------------------------------------
def _route_concept6(self, ck, bench, name, ks):
    if bench not in BENCHES or not all(c.isalnum() or c in "_.-" for c in name) or not ks or len(ks) > 12 or not all(0 <= k < NF for k in ks):
        return None
    out = f"{ROOT}/routes_concept6/ck{ck}/{bench}/{name}_{'-'.join(map(str, ks))}.png"
    if os.path.exists(out):
        return out
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Circle, FancyBboxPatch
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgb
    from matplotlib.transforms import Affine2D
    s = next(x for x in self.load(ck, bench) if x["name"] == name)
    _, folder, sub, *_ = sub_for(ck, bench)
    info = json.load(open(f"{self.D.R}/ckpt{ck}/{sub}/samples_meta.json"))[str(s["i"])]
    yaw, xy, _ = self._gtpose(ck, bench)[name]
    h = np.deg2rad(np.array([yaw[k] - yaw[ks[0]] for k in ks])); N = len(ks)
    P = [np.zeros(2)]
    for n in range(N - 1):
        dist = float(np.linalg.norm(xy[ks[n + 1]] - xy[ks[n]])); step = 1.25 if dist >= 0.3 else 0.75
        a = (h[n] + h[n + 1]) / 2; P.append(P[-1] + step * np.array([math.sin(a), math.cos(a)]))
    P = np.array(P)
    ramp = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    def col(t):
        x = min(max(t, 0), 1) * (len(ramp) - 1); i = min(int(x), len(ramp) - 2); f = x - i
        return tuple((1 - f) * np.array(to_rgb(ramp[i])) + f * np.array(to_rgb(ramp[i + 1])))
    cols = [col(0.15 + 0.85 * n / max(N - 1, 1)) for n in range(N)]
    u0 = np.array([math.sin(h[0]), math.cos(h[0])]); u1 = np.array([math.sin(h[-1]), math.cos(h[-1])])
    Q = np.vstack([P[0] - 0.6 * u0, P, P[-1] + 0.75 * u1]); pts = [Q[0]]
    for i in range(1, len(Q) - 1):
        a0 = pts[-1]; c = Q[i]; b = (Q[i] + Q[i + 1]) / 2 if i < len(Q) - 2 else Q[i + 1]
        for t in np.linspace(0, 1, 40)[1:]:
            pts.append((1 - t) ** 2 * a0 + 2 * (1 - t) * t * c + t ** 2 * b)
    pts = np.array(pts); tt = np.linspace(0, 1, len(pts)); segs = np.stack([pts[:-1], pts[1:]], 1)
    # photo cards: GT frame thumbnails, alternating sides, perpendicular to the local direction
    CW, CH, OFF = 1.30, 1.30 * 222 / 384, 1.25
    cards = []
    for n, k in enumerate(ks):
        d = np.array([math.sin(h[n]), math.cos(h[n])]); perp = np.array([d[1], -d[0]]) * (1 if n % 2 == 0 else -1)
        c = P[n] + perp * OFF
        img = cv2.cvtColor(cut(cv2.imread(f"{info['seq_dir'].rstrip('/')}/images/{s['start'] + k:06d}.png"), None), cv2.COLOR_BGR2RGB)
        cards.append((c, img))
    allp = np.vstack([pts, P + 0.7, P - 0.7] + [np.array([c + [-CW / 2 - .1, -CH / 2 - .1], c + [CW / 2 + .1, CH / 2 + .1]]) for c, _ in cards])
    x0, x1 = allp[:, 0].min() - 0.15, allp[:, 0].max() + 0.15; y0, y1 = allp[:, 1].min() - 0.15, allp[:, 1].max() + 0.15
    W_, H_ = x1 - x0, y1 - y0; fig, ax = plt.subplots(figsize=(3.4, 3.4 * H_ / W_), dpi=260)
    ax.add_patch(FancyBboxPatch((x0, y0), W_, H_, boxstyle="round,pad=0,rounding_size=0.3", fc="#f5f8fc", ec="#e2e8f0", lw=0.8, zorder=0))
    # soft view cones
    for n in range(N):
        p = P[n]; c = cols[n]; Lc, half = 0.75, math.radians(46.5)
        for j in range(16):
            r0, r1 = Lc * j / 16, Lc * (j + 1) / 16
            arc = lambda r: [p + r * np.array([math.sin(a), math.cos(a)]) for a in np.linspace(h[n] - half, h[n] + half, 18)]
            ax.add_patch(Polygon(arc(r1) + arc(r0)[::-1], closed=True, fc=c, ec="none", alpha=0.32 * (1 - j / 16) ** 1.5, zorder=1))
    # path: shadow, white edge, opaque gradient core (round caps -> no seams)
    ax.plot(pts[:, 0] + .03, pts[:, 1] - .045, "-", color="black", alpha=0.08, lw=8, solid_capstyle="round", zorder=2)
    ax.plot(pts[:, 0], pts[:, 1], "-", color="white", lw=7.5, solid_capstyle="round", zorder=3)
    ax.add_collection(LineCollection(segs, colors=[col(0.15 + 0.85 * t) for t in tt[:-1]], linewidths=4.6, capstyle="round", joinstyle="round", zorder=4))
    v = pts[-1] - pts[-8]; v /= np.linalg.norm(v) + 1e-9; nv = np.array([-v[1], v[0]]); sA = 0.24; tip = pts[-1] + v * 0.06
    ax.add_patch(Polygon([tip + v * sA, tip - v * sA * .4 + nv * sA * .75, tip - v * sA * .4 - nv * sA * .75], closed=True, fc=col(1.0), ec="white", lw=1.4, zorder=5, joinstyle="round"))
    ax.add_patch(Circle(Q[0], 0.09, fc="white", ec=col(0.15), lw=2, zorder=5))
    # connectors + cards
    for n, (c, img) in enumerate(cards):
        ax.plot([P[n][0], c[0]], [P[n][1], c[1]], "-", color=cols[n], lw=0.9, alpha=0.7, zorder=5.5)
        ax.add_patch(FancyBboxPatch((c[0] - CW / 2 + .03, c[1] - CH / 2 - .05), CW, CH, boxstyle="round,pad=0.04,rounding_size=0.07", fc=(0, 0, 0, .14), ec="none", zorder=6))
        ax.add_patch(FancyBboxPatch((c[0] - CW / 2, c[1] - CH / 2), CW, CH, boxstyle="round,pad=0.04,rounding_size=0.07", fc="white", ec=(*cols[n], 1), lw=1.2, zorder=6.5))
        ax.imshow(img, extent=(c[0] - CW / 2, c[0] + CW / 2, c[1] - CH / 2, c[1] + CH / 2), zorder=7, interpolation="lanczos")
        bc = c + np.array([-CW / 2 + 0.02, CH / 2 - 0.02])
        ax.add_patch(Circle(bc, 0.15, fc=cols[n], ec="white", lw=1.5, zorder=8))
        ax.text(*bc, str(n + 1), ha="center", va="center", fontsize=6.8, color="white", fontweight="bold", zorder=9)
    # camera glyphs at the waypoints (body + lens), rotated to the heading
    for n in range(N):
        tr = Affine2D().rotate(-h[n]).translate(*P[n]) + ax.transData
        ax.add_patch(Circle(P[n] + [.02, -.03], 0.2, color=(0, 0, 0, .15), zorder=9))
        body = FancyBboxPatch((-0.16, -0.12), 0.32, 0.22, boxstyle="round,pad=0,rounding_size=0.05", fc=cols[n], ec="white", lw=1.3, zorder=10, transform=tr)
        lens = Polygon([(-0.08, 0.10), (0.08, 0.10), (0.13, 0.22), (-0.13, 0.22)], closed=True, fc=cols[n], ec="white", lw=1.3, zorder=10, transform=tr)
        dot = Circle((0, -0.01), 0.055, fc="white", ec="none", zorder=11, transform=tr)
        ax.add_patch(lens); ax.add_patch(body); ax.add_patch(dot)
    ax.set_xlim(x0 - 0.02, x1 + 0.02); ax.set_ylim(y0 - 0.02, y1 + 0.02); ax.set_aspect("equal"); ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1); os.makedirs(os.path.dirname(out), exist_ok=True); fig.savefig(out, transparent=True); plt.close(fig)
    return out


Cherry.route = _route_concept6
_route_concept = _route_concept6


# ---- concept route v7 (rev 12): a road instead of an arrow ---------------------------------------------------------
def _route_concept7(self, ck, bench, name, ks):
    if bench not in BENCHES or not all(c.isalnum() or c in "_.-" for c in name) or not ks or len(ks) > 12 or not all(0 <= k < NF for k in ks):
        return None
    out = f"{ROOT}/routes_concept7/ck{ck}/{bench}/{name}_{'-'.join(map(str, ks))}.png"
    if os.path.exists(out):
        return out
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Circle, FancyBboxPatch
    from matplotlib.colors import to_rgb
    yaw, xy, _ = self._gtpose(ck, bench)[name]
    h = np.deg2rad(np.array([yaw[k] - yaw[ks[0]] for k in ks])); N = len(ks)
    P = [np.zeros(2)]
    for n in range(N - 1):
        dist = float(np.linalg.norm(xy[ks[n + 1]] - xy[ks[n]])); step = 1.0 if dist >= 0.3 else 0.6
        a = (h[n] + h[n + 1]) / 2; P.append(P[-1] + step * np.array([math.sin(a), math.cos(a)]))
    P = np.array(P)
    ramp = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    def col(t):
        x = min(max(t, 0), 1) * (len(ramp) - 1); i = min(int(x), len(ramp) - 2); f = x - i
        return tuple((1 - f) * np.array(to_rgb(ramp[i])) + f * np.array(to_rgb(ramp[i + 1])))
    cols = [col(0.2 + 0.8 * n / max(N - 1, 1)) for n in range(N)]
    u0 = np.array([math.sin(h[0]), math.cos(h[0])]); u1 = np.array([math.sin(h[-1]), math.cos(h[-1])])
    Q = np.vstack([P[0] - 0.7 * u0, P, P[-1] + 1.1 * u1]); pts = [Q[0]]
    for i in range(1, len(Q) - 1):
        a0 = pts[-1]; c = Q[i]; b = (Q[i] + Q[i + 1]) / 2 if i < len(Q) - 2 else Q[i + 1]
        for t in np.linspace(0, 1, 40)[1:]:
            pts.append((1 - t) ** 2 * a0 + 2 * (1 - t) * t * c + t ** 2 * b)
    pts = np.array(pts)
    tang = np.gradient(pts, axis=0); tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9; nrm = np.stack([tang[:, 1], -tang[:, 0]], 1)
    W = 0.30                                                            # road half-width
    L_, R_ = pts + nrm * W, pts - nrm * W
    allp = np.vstack([L_, R_] + [P[n] + 0.85 * np.array([math.sin(a), math.cos(a)]) for n in range(N) for a in (h[n] - .8, h[n], h[n] + .8)])
    x0, x1 = allp[:, 0].min() - 0.3, allp[:, 0].max() + 0.3; y0, y1 = allp[:, 1].min() - 0.25, allp[:, 1].max() + 0.25
    fig, ax = plt.subplots(figsize=(2.9, 2.9 * (y1 - y0) / (x1 - x0)), dpi=260)
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.28", fc="#eef3ea", ec="#dfe6da", lw=0.8, zorder=0))
    # view beams under the road
    for n in range(N):
        p = P[n]; c = cols[n]; Lc, half = 0.85, math.radians(46.5)
        for j in range(16):
            r0, r1 = Lc * j / 16, Lc * (j + 1) / 16
            arc = lambda r: [p + r * np.array([math.sin(a), math.cos(a)]) for a in np.linspace(h[n] - half, h[n] + half, 18)]
            ax.add_patch(Polygon(arc(r1) + arc(r0)[::-1], closed=True, fc=c, ec="none", alpha=0.30 * (1 - j / 16) ** 1.4, zorder=1))
    # road: soft shadow, curb, asphalt, fading tail
    ax.add_patch(Polygon(np.vstack([L_ + [.03, -.05], R_[::-1] + [.03, -.05]]), closed=True, fc=(0, 0, 0, .08), ec="none", zorder=2))
    ax.add_patch(Polygon(np.vstack([pts + nrm * (W + .05), (pts - nrm * (W + .05))[::-1]]), closed=True, fc="#ffffff", ec="none", zorder=2.5))
    ax.add_patch(Polygon(np.vstack([L_, R_[::-1]]), closed=True, fc="#6b7280", ec="none", zorder=3))
    ax.plot(L_[:, 0], L_[:, 1], "-", color="#e5e7eb", lw=0.9, zorder=3.2); ax.plot(R_[:, 0], R_[:, 1], "-", color="#e5e7eb", lw=0.9, zorder=3.2)
    ax.plot(pts[:, 0], pts[:, 1], color="white", lw=1.1, dashes=(5, 4), zorder=3.4)
    nt = max(int(0.12 * len(pts)), 8)                                   # fade the road into the background at the far end
    for j in range(nt):
        i = len(pts) - nt + j; a = (j + 1) / nt
        seg = np.vstack([L_[i - 1:i + 1] + 0, R_[i - 1:i + 1][::-1]])
        ax.add_patch(Polygon(np.vstack([pts[i - 1:i + 1] + nrm[i - 1:i + 1] * (W + .07), (pts[i - 1:i + 1] - nrm[i - 1:i + 1] * (W + .07))[::-1]]),
                             closed=True, fc="#eef3ea", ec="none", alpha=a ** 1.2, zorder=3.6))
    # painted road arrow between the last two markers
    ia = int(np.argmin(np.linalg.norm(pts - (P[-1] + 0.5 * u1), axis=1))); c0 = pts[ia]; d = tang[ia]; nv = nrm[ia]; s = 0.22
    arrow = [c0 + d * s, c0 + nv * s * .55, c0 + nv * s * .2, c0 - d * s * 1.1 + nv * s * .2, c0 - d * s * 1.1 - nv * s * .2, c0 - nv * s * .2, c0 - nv * s * .55]
    ax.add_patch(Polygon(arrow, closed=True, fc="white", ec="none", alpha=0.95, zorder=3.5))
    # numbered waypoints on the road
    for n in range(N):
        p = P[n]
        ax.add_patch(Circle(p + [.02, -.03], 0.17, color=(0, 0, 0, .2), zorder=6))
        ax.add_patch(Circle(p, 0.16, fc=cols[n], ec="white", lw=1.6, zorder=7))
        ax.text(*p, str(n + 1), ha="center", va="center", fontsize=7.2, color="white", fontweight="bold", zorder=8)
    ax.set_xlim(x0 - 0.02, x1 + 0.02); ax.set_ylim(y0 - 0.02, y1 + 0.02); ax.set_aspect("equal"); ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1); os.makedirs(os.path.dirname(out), exist_ok=True); fig.savefig(out, transparent=True); plt.close(fig)
    return out


Cherry.route = _route_concept7
_route_concept = _route_concept7


# ---- style gallery (rev 13): four conceptual styles, chosen via cache/cherry/route_style.txt ------------------------
STYLE_FILE = f"{ROOT}/route_style.txt"


def _schem(self, ck, bench, name, ks, step_move=1.0, step_turn=0.6):
    yaw, xy, _ = self._gtpose(ck, bench)[name]
    h = np.deg2rad(np.array([yaw[k] - yaw[ks[0]] for k in ks])); N = len(ks)
    P = [np.zeros(2)]
    for n in range(N - 1):
        dist = float(np.linalg.norm(xy[ks[n + 1]] - xy[ks[n]])); st_ = step_move if dist >= 0.3 else step_turn
        a = (h[n] + h[n + 1]) / 2; P.append(P[-1] + st_ * np.array([math.sin(a), math.cos(a)]))
    P = np.array(P); u0 = np.array([math.sin(h[0]), math.cos(h[0])]); u1 = np.array([math.sin(h[-1]), math.cos(h[-1])])
    Q = np.vstack([P[0] - 0.55 * u0, P, P[-1] + 0.7 * u1]); pts = [Q[0]]
    for i in range(1, len(Q) - 1):
        a0 = pts[-1]; c = Q[i]; b = (Q[i] + Q[i + 1]) / 2 if i < len(Q) - 2 else Q[i + 1]
        for t in np.linspace(0, 1, 40)[1:]:
            pts.append((1 - t) ** 2 * a0 + 2 * (1 - t) * t * c + t ** 2 * b)
    return h, P, np.array(pts)


def _draw_style(self, ax, style, ck, bench, name, ks):
    from matplotlib.patches import Polygon, Circle, FancyBboxPatch
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgb
    ramp = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    def col(t):
        x = min(max(t, 0), 1) * (len(ramp) - 1); i = min(int(x), len(ramp) - 2); f = x - i
        return tuple((1 - f) * np.array(to_rgb(ramp[i])) + f * np.array(to_rgb(ramp[i + 1])))
    h, P, pts = self._schem(ck, bench, name, ks); N = len(ks)
    cols = [col(0.2 + 0.8 * n / max(N - 1, 1)) for n in range(N)]; tt = np.linspace(0, 1, len(pts))
    segs = np.stack([pts[:-1], pts[1:]], 1); half = math.radians(46.5)
    dirv = lambda a: np.array([math.sin(a), math.cos(a)])
    def num(p, n, r=0.15, fs=7):
        ax.add_patch(Circle(p, r, fc=cols[n], ec="white", lw=1.5, zorder=8)); ax.text(*p, str(n + 1), ha="center", va="center", fontsize=fs, color="white", fontweight="bold", zorder=9)
    if style == "A":     # minimal line + faint view arcs
        ax.add_collection(LineCollection(segs, colors=[col(0.2 + 0.8 * t) for t in tt[:-1]], linewidths=2.6, capstyle="round", zorder=3))
        for n in range(N):
            p = P[n]
            arc = [p + 0.62 * dirv(a) for a in np.linspace(h[n] - half, h[n] + half, 24)]
            ax.add_patch(Polygon([p] + arc, closed=True, fc=cols[n], ec="none", alpha=0.10, zorder=1))
            ax.plot(*np.array(arc).T, "-", color=cols[n], lw=1.0, alpha=0.6, zorder=2, solid_capstyle="round")
            num(p, n, 0.13, 6.5)
    elif style == "B":   # dotted trail + heading ticks
        cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        for d in np.arange(0, cum[-1], 0.16):
            i = np.searchsorted(cum, d); t = d / cum[-1]
            ax.add_patch(Circle(pts[min(i, len(pts) - 1)], 0.032 + 0.012 * t, fc=col(0.2 + 0.8 * t), ec="none", alpha=0.9, zorder=3))
        for n in range(N):
            p = P[n]; e = p + 0.42 * dirv(h[n])
            ax.annotate("", e, p, arrowprops=dict(arrowstyle="-|>", color=cols[n], lw=1.3, mutation_scale=8), zorder=4)
            ax.add_patch(Circle(p, 0.17, fc="white", ec=cols[n], lw=2.0, zorder=8))
            ax.text(*p, str(n + 1), ha="center", va="center", fontsize=7, color=cols[n], fontweight="bold", zorder=9)
    elif style == "C":   # soft ribbon + pills
        tang = np.gradient(pts, axis=0); tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9; nrm = np.stack([tang[:, 1], -tang[:, 0]], 1)
        w = 0.26   # one polygon -> no seams; gradient via an image clipped to the ribbon
        rib = Polygon(np.vstack([pts + nrm * w, (pts - nrm * w)[::-1]]), closed=True, fc="none", ec="none")
        ax.add_patch(rib)
        ys = np.linspace(pts[:, 1].min() - w, pts[:, 1].max() + w, 256)[:, None]
        grad = np.array([col(0.2 + 0.8 * t) for t in np.linspace(0, 1, 256)])[:, None, :].repeat(2, 1)
        im = ax.imshow(np.concatenate([grad, np.full((256, 2, 1), 0.28)], 2), origin="lower", aspect="auto", zorder=1,
                       extent=(pts[:, 0].min() - w, pts[:, 0].max() + w, ys.min(), ys.max()), interpolation="bilinear")
        im.set_clip_path(rib)
        ax.plot(pts[:, 0], pts[:, 1], "-", color="white", lw=1.2, alpha=0.9, zorder=2)
        for n in range(N):
            p = P[n]
            ax.text(*p, f" {n + 1} ", ha="center", va="center", fontsize=7, color="white", fontweight="bold", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.9", fc=cols[n], ec="white", lw=1.3))
    elif style == "E":   # C ribbon + D frustums (pills on top)
        _draw_style(self, ax, "C", ck, bench, name, ks)
        for n in range(N):
            p = P[n]; d = dirv(h[n]); nv = np.array([d[1], -d[0]]); near, far, wn, wf = 0.22, 0.62, 0.10, 0.48
            tr = [p + d * near + nv * wn, p + d * far + nv * wf, p + d * far - nv * wf, p + d * near - nv * wn]
            ax.add_patch(Polygon(tr, closed=True, fc=cols[n], ec="none", alpha=0.16, zorder=3))
            ax.add_patch(Polygon(tr, closed=True, fc="none", ec=cols[n], lw=1.3, zorder=3.5, joinstyle="round"))
    elif style in ("F", "G"):   # polished ribbon + forward frustums (G: white light-beam frustums above the ribbon)
        tang = np.gradient(pts, axis=0); tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9; nrm = np.stack([tang[:, 1], -tang[:, 0]], 1)
        M = len(pts); taper = np.ones(M); k0 = int(M * 0.86); taper[k0:] = np.linspace(1, 0.08, M - k0) ** 0.9   # pointed tip = direction
        def ribbon(w, alpha, z, dx=0.0, dy=0.0, solid=None):
            poly = Polygon(np.vstack([pts + nrm * (w * taper)[:, None], (pts - nrm * (w * taper)[:, None])[::-1]]) + [dx, dy], closed=True, fc="none", ec="none")
            ax.add_patch(poly)
            if solid is not None:
                poly.set_facecolor(solid); poly.set_alpha(alpha); poly.set_zorder(z); return
            ext = (pts[:, 0].min() - w, pts[:, 0].max() + w, pts[:, 1].min() - w, pts[:, 1].max() + w)
            # gradient along the path: map each pixel row/col to the nearest path parameter (cheap: project on the start->end axis)
            gx, gy = np.meshgrid(np.linspace(ext[0], ext[1], 160), np.linspace(ext[2], ext[3], 220))
            d2 = (gx[..., None] - pts[None, None, ::6, 0]) ** 2 + (gy[..., None] - pts[None, None, ::6, 1]) ** 2
            tpar = np.argmin(d2, axis=2) / max(len(pts[::6]) - 1, 1)
            rgba = np.array([[(*col(0.2 + 0.8 * t), alpha) for t in row] for row in tpar])
            im = ax.imshow(rgba, origin="lower", extent=ext, zorder=z, interpolation="bilinear", aspect="auto"); im.set_clip_path(poly)
        rw = 0.22 if style == "F" else 0.17
        ribbon(rw + 0.02, 0.10, 0.8, 0.035, -0.05, solid="black")      # soft drop shadow
        ribbon(rw + 0.12, 0.16, 0.9)                                    # outer glow
        ribbon(rw, 0.85, 1.0)                                           # ribbon body
        ax.plot(pts[:int(M * 0.9), 0], pts[:int(M * 0.9), 1], color="white", lw=0.9, alpha=0.85, dashes=(4, 3), zorder=1.2)
        for n in range(N):                                              # forward frustum: apex at the waypoint, fading fill
            p = P[n]; d = dirv(h[n]); nv = np.array([d[1], -d[0]]); near, far, wn, wf = 0.20, 0.80, 0.07, 0.36
            for j in range(14):
                f0, f1 = near + (far - near) * j / 14, near + (far - near) * (j + 1) / 14
                w0, w1 = wn + (wf - wn) * j / 14, wn + (wf - wn) * (j + 1) / 14
                fc_, a_ = (cols[n], 0.36) if style == "F" else ("white", 0.62)
                ax.add_patch(Polygon([p + d * f0 + nv * w0, p + d * f1 + nv * w1, p + d * f1 - nv * w1, p + d * f0 - nv * w0], closed=True,
                                     fc=fc_, ec="none", alpha=a_ * (1 - j / 14) ** 1.2, zorder=2 if style == "F" else 3))
            ec_, z_ = (cols[n], 2.2) if style == "F" else (tuple(np.array(cols[n]) * 0.8), 3.2)
            for sg in (1, -1):
                ax.plot(*np.array([p + d * near + sg * nv * wn, p + d * far + sg * nv * wf]).T, "-", color=ec_, lw=1.1, alpha=0.85, zorder=z_, solid_capstyle="round")
            ax.plot(*np.array([p + d * far + nv * wf, p + d * far - nv * wf]).T, "-", color=ec_, lw=1.0, alpha=0.30, zorder=z_, solid_capstyle="round")
        for n in range(N):
            p = P[n]
            ax.text(p[0] + .02, p[1] - .03, f" {n + 1} ", ha="center", va="center", fontsize=7, color=(0, 0, 0, 0), zorder=7,
                    bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.9", fc=(0, 0, 0, .18), ec="none"))
            ax.text(*p, f" {n + 1} ", ha="center", va="center", fontsize=7, color="white", fontweight="bold", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.9", fc=cols[n], ec="white", lw=1.4))
    elif style == "D":   # camera frustums along a faint dashed path
        ax.plot(pts[:, 0], pts[:, 1], color="#b6bec9", lw=1.2, dashes=(3, 3), zorder=1)
        for n in range(N):
            p = P[n]; d = dirv(h[n]); nv = np.array([d[1], -d[0]]); near, far, wn, wf = 0.10, 0.55, 0.08, 0.50
            tr = [p + d * near + nv * wn, p + d * far + nv * wf, p + d * far - nv * wf, p + d * near - nv * wn]
            ax.add_patch(Polygon(tr, closed=True, fc=cols[n], ec=cols[n], lw=1.4, alpha=0.18, zorder=2))
            ax.add_patch(Polygon(tr, closed=True, fc="none", ec=cols[n], lw=1.4, zorder=3, joinstyle="round"))
            for q in (tr[0], tr[3]):
                ax.plot([p[0], q[0]], [p[1], q[1]], "-", color=cols[n], lw=1.0, zorder=3)
            num(p, n, 0.12, 6.2)
    return P, pts


def _route_style(self, ck, bench, name, ks, style=None):
    if bench not in BENCHES or not all(c.isalnum() or c in "_.-" for c in name) or not ks or len(ks) > 12 or not all(0 <= k < NF for k in ks):
        return None
    style = style or (open(STYLE_FILE).read().strip() if os.path.exists(STYLE_FILE) else "D")
    out = f"{ROOT}/routes_style{style}/ck{ck}/{bench}/{name}_{'-'.join(map(str, ks))}.png"
    if os.path.exists(out):
        return out
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(2.4, 3.0), dpi=260)
    P, pts = _draw_style(self, ax, style, ck, bench, name, ks)
    if style != "H":
        allp = np.vstack([pts, P + 0.65, P - 0.65]); ax.set_xlim(allp[:, 0].min(), allp[:, 0].max()); ax.set_ylim(pts[:, 1].min() - .15, allp[:, 1].max())
        ax.set_aspect("equal", adjustable="datalim"); ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1); os.makedirs(os.path.dirname(out), exist_ok=True); fig.savefig(out, transparent=True); plt.close(fig)
    return out


def _gallery(self, ck, items):
    """one PNG: rows = styles A-D, columns = the given (bench, name, ks) items."""
    out = f"{ROOT}/style_gallery.png"
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    styles = getattr(self, "_gallery_styles", "ABCD")
    fig, axes = plt.subplots(len(styles), len(items), figsize=(2.3 * len(items) + 0.6, len(styles) * 2.6), dpi=170)
    for r, st_ in enumerate(styles):
        for c, (b, n, ks) in enumerate(items):
            ax = axes[r][c] if len(styles) > 1 else axes[c]; P, pts = _draw_style(self, ax, st_, ck, b, n, ks)
            if st_ != "H":
                allp = np.vstack([pts, P + 0.65, P - 0.65]); ax.set_xlim(allp[:, 0].min(), allp[:, 0].max()); ax.set_ylim(pts[:, 1].min() - .15, allp[:, 1].max())
                ax.set_aspect("equal", adjustable="datalim"); ax.axis("off")
            if c == 0:
                ax.text(-0.08, 0.5, {"E": "C+D", "F": "C+D v2", "G": "C+D v3", "H": "3D"}.get(st_, st_), transform=ax.transAxes, fontsize=18, fontweight="bold", color="#33415c", ha="right", va="center")
    fig.tight_layout(); fig.savefig(out, facecolor="white"); plt.close(fig)
    return out


Cherry._schem = _schem
Cherry.route = _route_style
_route_concept = _route_style
Cherry.gallery = _gallery


# ---- 3D oblique style (rev 14) ---------------------------------------------------------------------------------------
def _draw_3d(self, ax, ck, bench, name, ks, el_deg=45.0):
    """perspective view from behind/above the start: ground grid, gradient ribbon path on the ground, a camera
    frustum (pyramid) hovering over each selected waypoint with a stem to the ground, numbered badges."""
    from matplotlib.patches import Polygon, Circle, Ellipse
    from matplotlib.colors import to_rgb
    ramp = ["#f28bb4", "#e486c0", "#d384cb", "#bf84d5", "#a985dd", "#9387e2"]   # pink -> lavender
    def col(t):
        x = min(max(t, 0), 1) * (len(ramp) - 1); i = min(int(x), len(ramp) - 2); f = x - i
        return tuple((1 - f) * np.array(to_rgb(ramp[i])) + f * np.array(to_rgb(ramp[i + 1])))
    h, P, pts = self._schem(ck, bench, name, ks, step_move=1.9, step_turn=1.2); N = len(ks)
    cols = [col(n / max(N - 1, 1)) for n in range(N)]
    # camera for the drawing
    el = math.radians(el_deg); C = np.array([pts[:, 0].mean(), pts[:, 1].mean(), 0.0])
    fwd = np.array([0, math.cos(el), -math.sin(el)]); up = np.array([0, math.sin(el), math.cos(el)]); rt = np.array([1.0, 0, 0])
    span = max(np.ptp(pts[:, 1]), np.ptp(pts[:, 0]), 1.0); E = C - fwd * (1.05 * span + 3.5)
    def pr(p3):
        p3 = np.atleast_2d(p3); q = p3 - E; z = q @ fwd
        return np.stack([(q @ rt) / z, (q @ up) / z], 1) * 6.0
    G = lambda xy, z=0.0: np.column_stack([np.atleast_2d(xy), np.full(len(np.atleast_2d(xy)), z)])
    # ground: soft plane + perspective grid fading with distance
    gx0, gx1 = pts[:, 0].min() - 6, pts[:, 0].max() + 6; gy0, gy1 = pts[:, 1].min() - 2.5, pts[:, 1].max() + 4
    plane = pr(G(np.array([[gx0, gy0], [gx1, gy0], [gx1, gy1], [gx0, gy1]])))
    ax.add_patch(Polygon(plane, closed=True, fc="#f3f6fa", ec="none", zorder=0))
    for gxv in np.arange(math.ceil(gx0 * 2) / 2, gx1 + 1e-6, 0.5):
        ax.plot(*pr(G(np.array([[gxv, gy0], [gxv, gy1]]))).T, "-", color="#dde3ec", lw=0.6, zorder=0.1)
    for gyv in np.arange(math.ceil(gy0 * 2) / 2, gy1 + 1e-6, 0.5):
        ax.plot(*pr(G(np.array([[gx0, gyv], [gx1, gyv]]))).T, "-", color="#dde3ec", lw=0.6, zorder=0.1,
                alpha=float(np.clip(1.15 - (gyv - pts[:, 1].min()) / (np.ptp(pts[:, 1]) + 3), 0.25, 1)))
    # ribbon on the ground (opaque quads -> no seams), tapered tip
    tang = np.gradient(pts, axis=0); tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9; nrm = np.stack([tang[:, 1], -tang[:, 0]], 1)
    M = len(pts); taper = np.ones(M); k0 = int(M * 0.85); taper[k0:] = np.linspace(1, 0.05, M - k0); W = 0.26
    Lp, Rp = pts + nrm * (W * taper)[:, None], pts - nrm * (W * taper)[:, None]
    ax.add_patch(Polygon(np.vstack([pr(G(Lp + [.06, -.06])), pr(G(Rp + [.06, -.06]))[::-1]]), closed=True, fc=(0, 0, 0, .07), ec="none", zorder=0.5))
    for i in range(M - 1):
        c = col(i / (M - 1))
        q = np.vstack([pr(G(Lp[i:i + 2])), pr(G(Rp[i:i + 2]))[::-1]])
        ax.add_patch(Polygon(q, closed=True, fc=c, ec=c, lw=0.4, zorder=1))
    ax.plot(*pr(G(pts[:int(M * 0.88)])).T, color="white", lw=0.8, dashes=(4, 3), alpha=0.9, zorder=1.2)
    # frustums, painter's order: far -> near
    hc, near, far, wf, vf = 0.75, 0.06, 1.25, 0.62, 0.38
    order = sorted(range(N), key=lambda n: -float((G(P[n])[0] - E) @ fwd))
    for n in order:
        d = np.array([math.sin(h[n]), math.cos(h[n]), 0.0]); r3 = np.array([d[1], -d[0], 0.0]); z3 = np.array([0, 0, 1.0])
        A = np.array([P[n][0], P[n][1], hc]); Bc = A + d * far
        cr = [Bc + r3 * wf + z3 * vf, Bc - r3 * wf + z3 * vf, Bc - r3 * wf - z3 * vf, Bc + r3 * wf - z3 * vf]
        c = np.array(cols[n])
        ax.plot(*pr(np.array([G(P[n])[0], A])).T, "-", color=tuple(c * .85), lw=0.9, alpha=0.6, zorder=2 + n * 0.01)        # stem
        pass   # (ground dot removed: read as noise on the ribbon)
        faces = [[A, cr[i], cr[(i + 1) % 4]] for i in range(4)]
        shade = [1.0, 0.85, 0.7, 0.92]                                   # top, left, bottom, right
        for f, sh in sorted(zip(faces, shade), key=lambda t: -float((np.mean(t[0], 0) - E) @ fwd)):
            ax.add_patch(Polygon(pr(np.array(f)), closed=True, fc=tuple(np.clip(c * sh + (1 - sh) * 0.15, 0, 1)), ec="none", alpha=0.30, zorder=3 + n * 0.01))
        base = pr(np.array(cr)); ax.add_patch(Polygon(base, closed=True, fc=(1, 1, 1, .35), ec=tuple(c * .8), lw=1.0, zorder=3.1 + n * 0.01))
        for k3 in cr:
            ax.plot(*pr(np.array([A, k3])).T, "-", color=tuple(c * .8), lw=1.0, zorder=3.2 + n * 0.01)
        ap = pr(A)[0]
        ax.text(ap[0] - 0.18 * math.cos(h[n]), ap[1] + 0.05, f"{n + 1}", ha="center", va="center", fontsize=6.5, color="white", fontweight="bold", zorder=7 + n * 0.01,
                bbox=dict(boxstyle="circle,pad=0.28", fc=tuple(c), ec="white", lw=1.2))
    # frame on the content (ribbon + frustums), the ground plane just bleeds out of the frame
    cont = [pr(G(Lp)), pr(G(Rp))]
    for n in range(N):
        d = np.array([math.sin(h[n]), math.cos(h[n]), 0.0]); r3 = np.array([d[1], -d[0], 0.0]); A = np.array([P[n][0], P[n][1], hc]); Bc = A + d * far
        cont.append(pr(np.array([A + [0, 0, 0.25], Bc + r3 * wf + [0, 0, vf], Bc - r3 * wf + [0, 0, vf], Bc + r3 * wf - [0, 0, vf], Bc - r3 * wf - [0, 0, vf]])))
    allp = np.vstack(cont); mx = 0.12 * np.ptp(allp[:, 0]) + 0.1; my = 0.10 * np.ptp(allp[:, 1]) + 0.1
    ax.set_xlim(allp[:, 0].min() - mx, allp[:, 0].max() + mx); ax.set_ylim(allp[:, 1].min() - my, allp[:, 1].max() + my)
    ax.set_aspect("equal"); ax.axis("off")


_draw_style_2d = _draw_style


def _draw_style_any(self, ax, style, ck, bench, name, ks):
    if style == "H":
        _draw_3d(self, ax, ck, bench, name, ks); h, P, pts = self._schem(ck, bench, name, ks); return P, pts
    return _draw_style_2d(self, ax, style, ck, bench, name, ks)


_draw_style = _draw_style_any


# ---- whole-sequence dominance ranking (rev 15) -----------------------------------------------------------------------
def _rank(self, ck, bench, n=5, gmin=6, ymax=35.0, allow_cut=False):
    """every sequence, ranked by how much ours dominates over the whole window (no frame thresholds):
    win_all = share of frames where ours beats all 7 baselines (LPIPS vs GT), dom = mean(best baseline - ours).
    Frames shown: spread picks among the win-all frames (continuity as in spread); filled with top-margin frames if short."""
    benches = BENCHES if bench == "all" else [bench]
    rows, nsamp = [], 0
    for b in benches:
        gtp = self._gtpose(ck, b); dyn = self.dyn(ck, b)
        for s in self.load(ck, b):
            nsamp += 1
            yaw, xy, cut_ = gtp[s["name"]]
            if cut_ and not allow_cut:
                continue
            lp = s["lp"]; o, base = lp[0], lp[1:]
            bestv = np.nanmin(np.where(np.isnan(base), np.inf, base), axis=0); fin = np.isfinite(o) & np.isfinite(bestv)
            marg = np.where(fin, bestv - o, np.nan); win = fin & (marg > 0)
            Q = [int(k) for k in np.nonzero(win)[0]]
            sel, g = chain_pick(Q, n, yaw, gmin, ymax)
            if sel is None:                                     # fill: best-margin frames, spread by a greedy min gap
                order = [int(k) for k in np.argsort(-np.nan_to_num(marg, nan=-9))]; sel = []
                for k in order:
                    if all(abs(k - q) >= gmin for q in sel):
                        sel.append(k)
                    if len(sel) == n:
                        break
                sel = sorted(sel); g = int(min(np.diff(sel))) if len(sel) > 1 else 0
            med = np.nanmedian(base, axis=0)
            fr = [{"k": k, "gt_frame": s["start"] + k, "ours_lp": round(float(o[k]), 4), "med_lp": round(float(med[k]), 4),
                   "best_lp": round(float(bestv[k]), 4), "nbeat": int(np.sum(base[:, k] > o[k])), "ours_ps": round(float(s["ps"][0][k]), 2),
                   "dyaw": round(float(yaw[k] - yaw[sel[0]]), 1), "dist": round(float(np.linalg.norm(xy[k] - xy[sel[0]])), 2)} for k in sel]
            dv = dyn.get(s["name"], {"img": 0, "yaw": 0, "path": 0})
            rows.append({"bench": b, "label": LABEL[b], "i": s["i"], "name": s["name"], "frames": fr, "n_ok": int(win.sum()), "min_gap": g,
                         "span": sel[-1] - sel[0], "yaw_steps": [round(float(yaw[b2] - yaw[a2]), 1) for a2, b2 in zip(sel[:-1], sel[1:])],
                         "yaw_range": round(float(np.ptp(yaw[sel])), 1),
                         "dom": round(float(np.nanmean(marg)), 4), "win_all": round(float(win.sum() / max(fin.sum(), 1)), 3),
                         "ours_lp": round(float(np.nanmean(o)), 4), "margin_best": round(float(np.nanmean(marg)), 4),
                         "margin_med": round(float(np.nanmean(med - o)), 4), "sample_ours_lp": round(float(np.nanmean(o)), 4),
                         "desc": _describe(yaw, xy, sel)["text"], **dv})
    rows.sort(key=lambda r: -r["dom"])
    return json.dumps({"ck": ck, "bench": bench, "n_samples": nsamp, "n_cut": 0, "rows": rows, "dyn_pct": {}}, separators=(",", ":"), ensure_ascii=False).encode()


Cherry.rank = _rank


# ---- "our criteria" score, no cut (rev 16) ---------------------------------------------------------------------------
def best_chain(score, n, yaw, gmin, ymax):
    """choose n frames (increasing) maximising the summed score, consecutive gap >= gmin and |yaw step| <= ymax."""
    K = len(score); NEG = -1e9
    dp = np.full((n, K), NEG); bk = np.full((n, K), -1, int); dp[0] = np.where(np.isfinite(score), score, NEG)
    for c in range(1, n):
        for i in range(K):
            if not np.isfinite(score[i]):
                continue
            best, arg = NEG, -1
            for j in range(0, i - gmin + 1):
                if dp[c - 1][j] > best and abs(yaw[i] - yaw[j]) <= ymax:
                    best, arg = dp[c - 1][j], j
            if arg >= 0:
                dp[c][i] = best + score[i]; bk[c][i] = arg
    i = int(np.argmax(dp[n - 1]))
    if dp[n - 1][i] <= NEG / 2:
        return None
    sel = [i]
    for c in range(n - 1, 0, -1):
        i = int(bk[c][i]); sel.append(i)
    return sel[::-1]


def _rank2(self, ck, bench, n=5, gmin=6, ymax=35.0, allow_cut=False, lam=0.5, dyn_w=0.003, kmin=0):
    benches = BENCHES if bench == "all" else [bench]
    rows, nsamp = [], 0
    allimg = [v["img"] for b in benches for v in self.dyn(ck, b).values()]; img_med = float(np.median(allimg)) if allimg else 0
    for b in benches:
        gtp = self._gtpose(ck, b); dyn = self.dyn(ck, b)
        for s in self.load(ck, b):
            nsamp += 1
            yaw, xy, cut_ = gtp[s["name"]]
            if cut_ and not allow_cut:
                continue
            lp = s["lp"]; o, base = lp[0], lp[1:]
            bestv = np.nanmin(np.where(np.isnan(base), np.inf, base), axis=0); med = np.nanmedian(base, axis=0)
            sc = np.where(np.isfinite(o) & np.isfinite(bestv), (bestv - o) - lam * o, np.nan)   # sequence score (ranking)
            # frames: ours close to GT AND even the closest baseline far from GT -> (min baseline LPIPS) - 1.5 * (ours LPIPS)
            selsc = sc.copy(); selsc[:kmin] = np.nan
            sel = best_chain(selsc, n, yaw, gmin, ymax)
            if sel is None:
                continue
            dv = dyn.get(s["name"], {"img": 0, "yaw": 0, "path": 0})
            fscore = float(np.mean(sc[sel])); total = fscore + dyn_w * (dv["img"] - img_med)
            fr = [{"k": k, "gt_frame": s["start"] + k, "ours_lp": round(float(o[k]), 4), "med_lp": round(float(med[k]), 4),
                   "best_lp": round(float(bestv[k]), 4), "nbeat": int(np.sum(base[:, k] > o[k])), "ours_ps": round(float(s["ps"][0][k]), 2),
                   "dyaw": round(float(yaw[k] - yaw[sel[0]]), 1), "dist": round(float(np.linalg.norm(xy[k] - xy[sel[0]])), 2)} for k in sel]
            fin = np.isfinite(o) & np.isfinite(bestv)
            rows.append({"bench": b, "label": LABEL[b], "i": s["i"], "name": s["name"], "frames": fr, "n_ok": int((fin & (bestv > o)).sum()),
                         "min_gap": int(min(np.diff(sel))), "span": sel[-1] - sel[0],
                         "yaw_steps": [round(float(yaw[b2] - yaw[a2]), 1) for a2, b2 in zip(sel[:-1], sel[1:])], "yaw_range": round(float(np.ptp(yaw[sel])), 1),
                         "score": round(total, 4), "frame_score": round(fscore, 4),
                         "dom": round(float(np.nanmean(np.where(fin, bestv - o, np.nan))), 4), "win_all": round(float((fin & (bestv > o)).sum() / max(fin.sum(), 1)), 3),
                         "ours_lp": round(float(np.mean(o[sel])), 4), "margin_best": round(float(np.mean(bestv[sel] - o[sel])), 4),
                         "margin_med": round(float(np.mean(med[sel] - o[sel])), 4), "sample_ours_lp": round(float(np.nanmean(o)), 4),
                         "desc": _describe(yaw, xy, sel)["text"], **dv})
    rows.sort(key=lambda r: -r["score"])
    return json.dumps({"ck": ck, "bench": bench, "n_samples": nsamp, "n_cut": 0, "rows": rows, "dyn_pct": {},
                       "params": {"lam": lam, "dyn_w": dyn_w, "img_med": img_med}}, separators=(",", ":"), ensure_ascii=False).encode()


Cherry.rank = _rank2


# ---- label-free grid for figures (rev 17) ----------------------------------------------------------------------------
GRID_GAP = 8


def _grid(self, ck, bench, name, ks, gap=GRID_GAP):
    """rows = selected frames, columns = GT | ours | 7 baselines (MODELS order); every tile = the common GT crop at 384x222;
    uniform white gutters, no text. -> path of the PNG (cache)."""
    s = next((x for x in self.load(ck, bench) if x["name"] == name), None)
    if s is None:
        return None
    _, folder, sub, *_ = sub_for(ck, bench)
    info = json.load(open(f"{self.D.R}/ckpt{ck}/{sub}/samples_meta.json"))[str(s["i"])]
    bmeta = {mid: json.load(open(f"{CMP_ROOT}/{mid}/{folder}/{name}/meta.json")).get("gt_crop_box_xyxy") for mid, _ in MODELS}
    cols = 2 + len(MODELS); rows = len(ks)
    G = np.full((rows * H + (rows - 1) * gap, cols * W + (cols - 1) * gap, 3), 255, np.uint8)
    for r, k in enumerate(ks):
        tiles = [cut(cv2.imread(f"{info['seq_dir'].rstrip('/')}/images/{s['start'] + k:06d}.png"), None),
                 self.frame(f"{self.D.R}/ckpt{ck}/{sub}/sample_{s['i']:02d}.mp4", k, None)]
        tiles += [self.frame(f"{CMP_ROOT}/{mid}/{folder}/{name}/video.mp4", k, bmeta[mid]) for mid, _ in MODELS]
        for c, t in enumerate(tiles):
            y, x = r * (H + gap), c * (W + gap); G[y:y + H, x:x + W] = t
    out = f"{ROOT}/grids/ck{ck}/{bench}/{name}_{'-'.join(map(str, ks))}.png"
    os.makedirs(os.path.dirname(out), exist_ok=True); cv2.imwrite(out, G)
    return out


_export_v6 = Cherry.export


def _export7(self, ck, bench, name, ks):
    root = _export_v6(self, ck, bench, name, ks)
    if root:
        import shutil
        g = _grid(self, ck, bench, name, ks)
        if g:
            shutil.copy(g, f"{root}/grid.png")
            m = json.load(open(f"{root}/meta.json"))
            m["grid_png"] = {"rows": [f"f{k:02d}" for k in ks], "cols": ["gt", f"ours_ck{ck}"] + [mid for mid, _ in MODELS],
                             "tile_px": [W, H], "gap_px": GRID_GAP, "crop": "common GT rows 108..404 for every model (as the LPIPS), no text"}
            json.dump(m, open(f"{root}/meta.json", "w"), indent=1, ensure_ascii=False)
    return root


Cherry.grid = _grid
Cherry.export = _export7


# ---- grid in the paper's table order, placeholders for missing methods (rev 18) --------------------------------------
GRID_ORDER = ["wan21_flf2v", "wan21_vace", "wan22_fun_inp", "vggtomega_seva", "vggtomega_vggtdiff",
              "nwm_nomad", "nomad_deepverse", "nomad_lingbot_oneshot", "nomad_egowm", "aether_planning", "ours", "gt"]
GRID_PLACEHOLDER = (217, 220, 225)     # light gray tile, no text (BGR)


def _grid2(self, ck, bench, name, ks, gap=GRID_GAP):
    s = next((x for x in self.load(ck, bench) if x["name"] == name), None)
    if s is None:
        return None
    _, folder, sub, *_ = sub_for(ck, bench)
    info = json.load(open(f"{self.D.R}/ckpt{ck}/{sub}/samples_meta.json"))[str(s["i"])]
    cols, rows = len(GRID_ORDER), len(ks)
    G = np.full((rows * H + (rows - 1) * gap, cols * W + (cols - 1) * gap, 3), 255, np.uint8)
    present = {}
    for c, mid in enumerate(GRID_ORDER):
        if mid in ("gt", "ours"):
            present[mid] = True; continue
        mp = f"{CMP_ROOT}/{mid}/{folder}/{name}"
        present[mid] = os.path.exists(f"{mp}/video.mp4")
    for r, k in enumerate(ks):
        for c, mid in enumerate(GRID_ORDER):
            if mid == "gt":
                t = cut(cv2.imread(f"{info['seq_dir'].rstrip('/')}/images/{s['start'] + k:06d}.png"), None)
            elif mid == "ours":
                t = self.frame(f"{self.D.R}/ckpt{ck}/{sub}/sample_{s['i']:02d}.mp4", k, None)
            elif present[mid]:
                mp = f"{CMP_ROOT}/{mid}/{folder}/{name}"
                t = self.frame(f"{mp}/video.mp4", k, json.load(open(f"{mp}/meta.json")).get("gt_crop_box_xyxy"))
            else:
                t = np.full((H, W, 3), GRID_PLACEHOLDER, np.uint8)
            y, x = r * (H + gap), c * (W + gap); G[y:y + H, x:x + W] = t
    out = f"{ROOT}/grids2/ck{ck}/{bench}/{name}_{'-'.join(map(str, ks))}.png"
    os.makedirs(os.path.dirname(out), exist_ok=True); cv2.imwrite(out, G)
    self._grid_present = present
    return out


def _export8(self, ck, bench, name, ks):
    root = _export_v6(self, ck, bench, name, ks)
    if root:
        import shutil
        g = _grid2(self, ck, bench, name, ks)
        if g:
            shutil.copy(g, f"{root}/grid.png")
            m = json.load(open(f"{root}/meta.json"))
            m["grid_png"] = {"rows": [f"f{k:02d}" for k in ks], "cols": GRID_ORDER,
                             "placeholder_cols": [mid for mid in GRID_ORDER if not self._grid_present.get(mid)],
                             "tile_px": [W, H], "gap_px": GRID_GAP, "crop": "common GT rows 108..404 for every model (as the LPIPS), no text"}
            json.dump(m, open(f"{root}/meta.json", "w"), indent=1, ensure_ascii=False)
    return root


Cherry.grid = _grid2
Cherry.export = _export8
