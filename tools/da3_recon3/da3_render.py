"""Stage 2 (/venv/main): recon_fig.py stills from the DA3 reconstructions of GT / ours / Aether (da3_run.py).
Each video: DA3 depth + DA3 camera, ONE scale per video s = median(GT depth / DA3 depth) over all frames (DA3-LARGE is
relative), same s on the camera translations; frame 0 anchored on the GT start pose (like ours' re-anchoring).
Drawing identical to recon_fig.render: 256x171 grid, stride 6, size 400, shared extent (p98 over all three clouds),
ONE pixel mask for all three (every depth < 20 m, no flying pixel > 0.05 in any), GT RGB on every cloud.
  -> <out>/recon/{gt,ours,aether}/azAAA.png  <out>/bev/{gt,ours,aether}.png  <out>/da3_recon.json"""
import json, math, os, sys
import cv2, numpy as np
sys.path.insert(0, "/root/project/evalwam/tools")
from recon_fig import splat, edge
from recon_orbit_rollout import c2w, unproj, NEAR, FAR, _C3
from aether_recon_fig import samp, SD, ST, Y0, Y1, GW, GH, MAXD, EDGE, STRIDE, SIZE, ELEV, AZ, C3H
OUT = sys.argv[1]; TAGS = ("gt", "ours", "aether")
gp = np.load(f"{SD}/poses.npy"); nd = np.load(f"{SD}/depth/{ST:06d}.npy").shape[0]
Z = {k: np.load(f"{OUT}/da3_{k}.npz") for k in TAGS}
TS = [int(t) for t in Z["gt"]["ts"]]; FR = [ST + t for t in TS]
dg = [samp(np.load(f"{SD}/depth/{f:06d}.npy").astype(np.float64), GH, GW, Y0 * nd / 512, Y1 * nd / 512) for f in FR]
G0 = c2w(gp[ST]) @ C3H
D, P, K, scl = {}, {}, {}, {}
for k in TAGS:
    z = Z[k]; h, w = z["depth"].shape[1:]
    d = [samp(z["depth"][i].astype(np.float64), GH, GW) for i in range(len(FR))]
    v = np.concatenate([(dg[i] / np.maximum(d[i], 1e-6))[(dg[i] > NEAR) & (dg[i] < FAR) & (d[i] > 0)] for i in range(len(FR))])
    s = float(np.median(v)); scl[k] = s
    D[k] = [s * x for x in d]
    C = z["c2w"].copy(); C[:, :3, 3] *= s
    A = G0 @ np.linalg.inv(C[0])
    P[k] = [A @ C[i] @ np.linalg.inv(C3H) for i in range(len(FR))]
    K[k] = [(z["K"][i][0, 0] * GW / w, z["K"][i][1, 1] * GH / h, z["K"][i][0, 2] * GW / w, z["K"][i][1, 2] * GH / h) for i in range(len(FR))]
M = []
for i in range(len(FR)):
    m = dg[i] > 0   # all True where GT is defined; DA3 depths gate the rest
    for k in TAGS:
        m &= (D[k][i] > NEAR) & (D[k][i] < MAXD) & ~edge(D[k][i], EDGE)
    M.append(m.ravel())
clouds = [np.concatenate([unproj(D[k][i], P[k][i], K[k][i])[M[i]][::STRIDE] for i in range(len(FR))]) for k in TAGS]
def rgb(i):
    im = cv2.imread(f"{SD}/images/{FR[i]:06d}.png")
    return np.stack([samp(im[..., c], GH, GW, Y0, Y1) for c in range(3)], -1).reshape(-1, 3)[M[i]][::STRIDE]
RGB = np.concatenate([rgb(i) for i in range(len(FR))]).astype(np.float64)
tracks = [np.stack([x[:3, 3] for x in P[k]]) for k in TAGS]
CT = [np.median(Q, 0) for Q in clouds]
EX = max(max(float(np.percentile(np.abs(np.concatenate([(Q - c)[:, 0], (Q - c)[:, 1]])), 98)) for Q, c in zip(clouds, CT)), 5.0)
S = SIZE; SX = int(round(S * 1.1)); SX += SX & 1
BY = np.linspace(-EX, EX, S + 1); BX = np.linspace(-EX * 1.1, EX * 1.1, SX + 1); lw = max(2, round(2 * S / 400))
os.makedirs(f"{OUT}/bev", exist_ok=True)
for j, tag in enumerate(TAGS):
    os.makedirs(f"{OUT}/recon/{tag}", exist_ok=True)
    Q = clouds[j] - CT[j]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        im = splat(Q[:, 1], Q[:, 0], RGB, BY, BX)
    TR = tracks[j] - CT[j]
    px = ((TR[:, 0] - BX[0]) / (BX[-1] - BX[0]) * SX).astype(np.int32); py = ((TR[:, 1] - BY[0]) / (BY[-1] - BY[0]) * S).astype(np.int32)
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
gt_true = np.stack([c2w(gp[f])[:3, 3] for f in FR])
# ---- overlay: each method's OWN camera (magenta) on its DA3 BEV (DA3 camera = yellow) ----
RD = "/root/project/egowam/results_syn/goalprog_mode_real6_nomr_hg0_grad0.5/ckpta7500_8750/lookout_rc4_viz"
raw = np.load(f"{RD}/rollout_43_pred_poses.npy")
A0 = c2w(gp[ST]) @ np.linalg.inv(c2w(raw[0]))          # recon_fig re-anchoring on the GT start pose
own = {"gt": gt_true, "ours": np.stack([(A0 @ c2w(raw[f - ST]))[:3, 3] for f in FR]),
       "aether": np.load(f"{os.path.dirname(OUT.rstrip('/'))}/aether_track_own.npy")}
def fit_yaw_scale(X, Y):
    """2D similarity ABOUT THE START (start stays on the shared start): yaw + scale minimising |s R (X - X0) - (Y - Y0)|^2
    in the BEV plane; z scaled by s. -> aligned X, s, yaw_deg"""
    u = (X[:, 0] - X[0, 0]) + 1j * (X[:, 1] - X[0, 1]); v = (Y[:, 0] - Y[0, 0]) + 1j * (Y[:, 1] - Y[0, 1])
    a = (np.conj(u) * v).sum() / max((np.abs(u) ** 2).sum(), 1e-12); w = a * u
    Z = np.stack([Y[0, 0] + w.real, Y[0, 1] + w.imag, Y[0, 2] + abs(a) * (X[:, 2] - X[0, 2])], 1)
    return Z, float(abs(a)), float(np.degrees(np.angle(a)))
ALIGN = {}
for j, tag in enumerate(TAGS):
    Z, s_, y_ = fit_yaw_scale(own[tag], tracks[j])
    ALIGN[tag] = {"scale": round(s_, 3), "yaw_deg": round(y_, 3),
                  "mean_dist_before_m": round(float(np.linalg.norm(own[tag] - tracks[j], axis=1).mean()), 3),
                  "mean_dist_after_m": round(float(np.linalg.norm(Z - tracks[j], axis=1).mean()), 3)}
    own[f"{tag}_aligned"] = Z
os.makedirs(f"{OUT}/bev_overlay", exist_ok=True)
for j, tag in list(enumerate(TAGS)) + [(j, f"{t}_aligned") for j, t in enumerate(TAGS)]:
    im = cv2.imread(f"{OUT}/bev/{tag.split('_')[0]}.png")
    TR = own[tag] - CT[j]
    px = ((TR[:, 0] - BX[0]) / (BX[-1] - BX[0]) * SX).astype(np.int32); py = ((TR[:, 1] - BY[0]) / (BY[-1] - BY[0]) * S).astype(np.int32)
    cv2.polylines(im, [np.stack([px, py], -1).reshape(-1, 1, 2)], False, (255, 0, 255), lw, cv2.LINE_AA)
    cv2.circle(im, (int(px[0]), int(py[0])), 2 * lw, (0, 255, 0), -1)
    cv2.imwrite(f"{OUT}/bev_overlay/{tag}.png", im)
pl = lambda T: float(np.linalg.norm(np.diff(T, axis=0), axis=1).sum())
info = {"scale_s": {k: round(scl[k], 3) for k in TAGS}, "extent_m": round(EX, 3), "n_points": len(clouds[0]),
        "path_m": {"gt_pose": round(pl(gt_true), 3), **{f"da3_{k}": round(pl(t), 3) for k, t in zip(TAGS, tracks)}},
        "end_err_vs_gt_pose_m": {f"da3_{k}": round(float(np.linalg.norm(t[-1] - gt_true[-1])), 3) for k, t in zip(TAGS, tracks)},
        "own_pose": {k: {"path_m": round(pl(own[k]), 3), "end_err_vs_gt_pose_m": round(float(np.linalg.norm(own[k][-1] - gt_true[-1])), 3),
                         "mean_dist_to_own_da3_m": round(float(np.linalg.norm(own[k] - t, axis=1).mean()), 3)}
                     for k, t in zip(TAGS, tracks)},
        "own_to_da3_yaw_scale_about_start": ALIGN}
np.savez(f"{OUT}/tracks_all.npz", fr=np.array(FR), **{f"da3_{k}": t for k, t in zip(TAGS, tracks)}, **{f"own_{k}": v for k, v in own.items()})
json.dump(info, open(f"{OUT}/da3_recon.json", "w"), indent=1); print(json.dumps(info))
