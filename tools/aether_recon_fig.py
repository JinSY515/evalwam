#!/usr/bin/env python3
"""recon_fig.py stills (orbit az 0/90/180/270 elev 22 + BEV) for the Aether long-horizon baseline, Trader Joe's rc4 clip.

Same drawing as recon_fig.render (same splat, stride 6, size 400, shared extent = p98 over both clouds, depth < 20 m,
flying-pixel 0.05 mask shared by both clouds, GT RGB on both, BEV camera path yellow + green start dot), on frames
t = 0, 4, ..., 304 (GT 4 + t), exactly the frames ours' rollout cloud uses.

  GT     = GT depth + GT pose, restricted to Aether's view (GT 512 frame rows 85..427 = Aether 720x480)
  AETHER = Aether's own generated disparity + Aether's own camera (raymap -> poses_c2w, per-frame focal), per chunk:
           * scale: one factor k_c per chunk, metric depth = k_c / disparity, k_c = median(GT depth * disparity) over
             the chunk's frames (Aether chunks are independent, scale-free; ours gets one global depth affine).
             The chunk's camera translations are multiplied by the same k_c so depth and camera stay consistent.
           * chaining: chunk 0 frame 0 is GT frame 4 -> anchored on the GT start pose (like ours' re-anchoring);
             chunk c>0 frame 0 IS the last generated frame of chunk c-1 -> placed on that frame's chained pose.
             No other GT pose is used.

  /venv/main/bin/python aether_recon_fig.py <aether_raw_chunks dir> <out root>
  -> <out>/recon/{gt,aether}/azAAA.png  <out>/bev/{gt,aether}.png  <out>/recon_aether.json
"""
import json, math, os, sys
import cv2, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recon_fig import splat, edge  # noqa: E402
from recon_orbit_rollout import c2w, unproj, NEAR, FAR, _C3  # noqa: E402  (path added by recon_fig)

SD = "/root/data/eval_bench_v2/lookout_turn_rollout_c4_cv_lh/aria/nav/Traderjoes_jan11_s00296"
ST, TS = 4, list(range(0, 305, 4))
Y0, Y1 = 85, 427                      # Aether view = GT rows 85..427 of the 512 frame (meta gt_crop_box_xyxy)
GW, GH = 256, 171                     # common pixel grid (GT band at half resolution; Aether 720x480 * 0.356)
MAXD, EDGE, STRIDE, SIZE, ELEV, AZ = 20.0, 0.05, 6, 400, 22.0, (0, 90, 180, 270)
C3H = np.eye(4); C3H[:3, :3] = _C3    # opencv camera axes -> GT body axes (unproj applies _C3)


def samp(a, H, W, y0=0.0, y1=None):
    """nearest-sample a (h, w) onto an H x W grid covering rows y0..y1 of a."""
    h, w = a.shape; y1 = h if y1 is None else y1
    yi = (y0 + (np.arange(H) + .5) * (y1 - y0) / H).astype(int).clip(0, h - 1)
    xi = ((np.arange(W) + .5) * w / W).astype(int).clip(0, w - 1)
    return a[yi][:, xi]


def main():
    RAW, OUT = sys.argv[1], sys.argv[2]
    gp = np.load(f"{SD}/poses.npy"); i0 = np.load(f"{SD}/intrinsics.npy").astype(float)
    CH = [json.load(open(f"{RAW}/chunk{c}/chunk_meta.json")) for c in range(8)]
    # GT frame -> (chunk, generated index); a boundary frame belongs to the chunk that generated it (the earlier one)
    where = {}
    for c, m in enumerate(CH):
        for i, g in enumerate(m["generated_to_gt_index"]):
            f = m["gt_first"] + i
            if f not in where:
                where[f] = (c, int(g))
    disp = [np.load(f"{RAW}/chunk{c}/disparity.npy", mmap_mode="r") for c in range(8)]
    pose = [np.load(f"{RAW}/chunk{c}/poses_c2w.npy").astype(np.float64) for c in range(8)]
    foc = [np.load(f"{RAW}/chunk{c}/focal_px.npy").astype(np.float64) for c in range(8)]
    nd = np.load(f"{SD}/depth/{ST:06d}.npy").shape[0]
    sg = GW / nd                       # GT: 512 -> 256 (same factor in y for the band)
    gi = (i0[0] * sg, i0[1] * sg, i0[2] * sg, (i0[3] - Y0 * nd / 512) * sg)
    dg = {f: samp(np.load(f"{SD}/depth/{f:06d}.npy").astype(np.float64), GH, GW, Y0 * nd / 512, Y1 * nd / 512)
          for f in range(ST, ST + 305)}
    da = {f: samp(np.asarray(disp[c][g], np.float64), GH, GW) for f, (c, g) in where.items()}

    # per-chunk depth scale from all GT frames of the chunk
    k = []
    for c in range(8):
        r = []
        for f, (cc, g) in where.items():
            if cc != c:
                continue
            m = (dg[f] > NEAR) & (dg[f] < FAR) & (da[f] > 0.03)
            r.append((dg[f] * da[f])[m])
        k.append(float(np.median(np.concatenate(r))))

    def S(P, s):
        Q = P.copy(); Q[:3, 3] *= s; return Q

    # chain the chunks in the GT world (opencv camera axes), only frame 0 anchored on GT
    W = {}
    prev = c2w(gp[ST]) @ C3H
    for c in range(8):
        A = prev @ np.linalg.inv(S(pose[c][0], k[c]))
        for g in range(len(pose[c])):
            W[(c, g)] = A @ S(pose[c][g], k[c])
        prev = W[(c, len(pose[c]) - 1)]

    FR = [ST + t for t in TS]
    dp = {f: k[where[f][0]] / np.clip(da[f], k[where[f][0]] / FAR, None) for f in FR}
    cg = {f: c2w(gp[f]) for f in FR}
    cp = {f: W[where[f]] @ np.linalg.inv(C3H) for f in FR}
    sa = GW / 720.0
    ai = {f: (foc[where[f][0]][where[f][1]] * sa, foc[where[f][0]][where[f][1]] * GH / 480.0, GW / 2, GH / 2) for f in FR}
    M = {}
    for f in FR:
        m = (dg[f] > NEAR) & (dg[f] < MAXD) & (dp[f] < MAXD) & (da[f] > 0)
        m &= ~edge(dg[f], EDGE) & ~edge(dp[f], EDGE)
        M[f] = m.ravel()
    clouds = [np.concatenate([unproj(dg[f], cg[f], gi)[M[f]][::STRIDE] for f in FR]),
              np.concatenate([unproj(dp[f], cp[f], ai[f])[M[f]][::STRIDE] for f in FR])]
    def rgb(f):   # GT colours on both clouds, as recon_fig (BGR, band, same grid)
        im = cv2.imread(f"{SD}/images/{f:06d}.png")
        return np.stack([samp(im[..., c], GH, GW, Y0, Y1) for c in range(3)], -1).reshape(-1, 3)[M[f]][::STRIDE]
    RGB = np.concatenate([rgb(f) for f in FR]).astype(np.float64)
    tracks = [np.stack([cg[f][:3, 3] for f in FR]), np.stack([cp[f][:3, 3] for f in FR])]

    # ---- drawing: identical to recon_fig.render ----
    CT = [np.median(P, 0) for P in clouds]
    EX = max(max(float(np.percentile(np.abs(np.concatenate([(P - c)[:, 0], (P - c)[:, 1]])), 98))
                 for P, c in zip(clouds, CT)), 5.0)
    Sz = SIZE; SX = int(round(Sz * 1.1)); SX += SX & 1
    BY = np.linspace(-EX, EX, Sz + 1); BX = np.linspace(-EX * 1.1, EX * 1.1, SX + 1)
    lw = max(2, round(2 * Sz / 400))
    for d in ("recon/gt", "recon/aether", "bev"):
        os.makedirs(f"{OUT}/{d}", exist_ok=True)
    for tag, j in (("gt", 0), ("aether", 1)):
        Q = clouds[j] - CT[j]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            im = splat(Q[:, 1], Q[:, 0], RGB, BY, BX)
        TR = tracks[j] - CT[j]
        px = ((TR[:, 0] - BX[0]) / (BX[-1] - BX[0]) * SX).astype(np.int32)
        py = ((TR[:, 1] - BY[0]) / (BY[-1] - BY[0]) * Sz).astype(np.int32)
        cv2.polylines(im, [np.stack([px, py], -1).reshape(-1, 1, 2)], False, (0, 255, 255), lw, cv2.LINE_AA)
        cv2.circle(im, (int(px[0]), int(py[0])), 2 * lw, (0, 255, 0), -1)
        cv2.imwrite(f"{OUT}/bev/{tag}.png", im)
        el = math.radians(ELEV)
        for a_ in AZ:
            th = math.radians(a_)
            fwd = np.array([math.cos(th) * math.cos(el), math.sin(th) * math.cos(el), math.sin(el)])
            rgt = np.cross(fwd, [0, 0, 1.0]); rgt /= np.linalg.norm(rgt); up2 = np.cross(rgt, fwd)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                im = splat(-(Q @ up2), Q @ rgt, RGB, BY, BX)
            cv2.imwrite(f"{OUT}/recon/{tag}/az{a_:03d}.png", im)

    gl = np.linalg.norm(np.diff(tracks[0], axis=0), axis=1).sum(); al = np.linalg.norm(np.diff(tracks[1], axis=0), axis=1).sum()
    info = {"chunk_depth_scale_k": [round(x, 3) for x in k], "extent_m": round(EX, 3), "n_points": [len(clouds[0]), len(clouds[1])],
            "path_m": {"gt": round(float(gl), 3), "aether": round(float(al), 3)},
            "end_err_m": round(float(np.linalg.norm(tracks[0][-1] - tracks[1][-1])), 3)}
    json.dump(info, open(f"{OUT}/recon_aether.json", "w"), indent=1)
    np.save(f"{OUT}/aether_track_own.npy", tracks[1])   # (77, 3) chained Aether camera positions, GT world, frames FR
    print(json.dumps(info))


if __name__ == "__main__":
    main()
