#!/usr/bin/env python3
"""Side-by-side panel video per sample of a flat results dir (LookOut / valbench layout):
    [ hist (last cond frame) | generated | GT | predicted depth | goal image ]
Inputs: <dir>/samples_meta.json (seq_dir, start_frame, num_frames, goal_frame), sample_XX.mp4,
optional sample_XX_pred_depth.npy (T_lat, h, w; disparity-affine -> per-sample min-max, TURBO).
Output: <dir>/<out_subdir>/sample_XX_panels.mp4 (+ sample_XX_panels.png: 5 time steps x panels).

  python3 tools/viz_panels.py --results_dir <dir> [--samples 0 3] [--size 320]
"""
import argparse, json, os, subprocess
import numpy as np, cv2


def read_mp4(p):
    cap, out = cv2.VideoCapture(p), []
    while True:
        ok, f = cap.read()
        if not ok:
            return out
        out.append(f)


def label(img, text, sub=None):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (6, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def depth_color(d, lo, hi, S):
    x = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)          # disparity-like: high = near
    return cv2.applyColorMap(cv2.resize((x * 255).astype(np.uint8), (S, S)), cv2.COLORMAP_TURBO)


def process(D, i, info, a, outdir):
    S = a.size; sd = info["seq_dir"]; s0 = int(info["start_frame"]); nf = int(info.get("num_frames", 77))
    g0 = int(info["goal_frame"])
    imgs = sorted(os.listdir(os.path.join(sd, "images")))
    rd = lambda j: cv2.resize(cv2.imread(os.path.join(sd, "images", imgs[min(j, len(imgs) - 1)])), (S, S))
    gen = read_mp4(os.path.join(D, f"sample_{i:02d}.mp4"))[:nf]
    n = len(gen)
    hist = label(rd(s0 - 1), "hist (cond)", f"frame {s0 - 1}")
    goal = label(rd(g0), "goal", f"frame {g0}")
    dp = os.path.join(D, f"sample_{i:02d}_pred_depth.npy")
    dep = np.load(dp) if os.path.exists(dp) else None
    if dep is not None:
        lo, hi = np.percentile(dep, 2), np.percentile(dep, 98)
    mp4 = os.path.join(outdir, f"sample_{i:02d}_panels.mp4")
    ncol = 5 if dep is not None else 4
    W, H = S * ncol, S
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                           "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", mp4],
                          stdin=subprocess.PIPE)
    want = set(np.linspace(0, n - 1, 5).round().astype(int).tolist()); snaps = []
    for t in range(n):
        p = [hist, label(cv2.resize(gen[t], (S, S)), "generated", f"t={t}"), label(rd(s0 + t), "GT", f"frame {s0 + t}")]
        if dep is not None:
            p.append(label(depth_color(dep[min((t + 3) // 4, len(dep) - 1)], lo, hi, S), "pred depth"))
        p.append(goal)
        fr = np.concatenate(p, 1); ff.stdin.write(fr.tobytes())
        if t in want:
            snaps.append(fr)
    ff.stdin.close(); ff.wait()
    cv2.imwrite(os.path.join(outdir, f"sample_{i:02d}_panels.png"), np.concatenate(snaps, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--samples", type=int, nargs="*", default=None)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--out_subdir", default="viz_panels")
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.results_dir, "samples_meta.json")))
    outdir = os.path.join(a.results_dir, a.out_subdir); os.makedirs(outdir, exist_ok=True)
    for i in (a.samples if a.samples else sorted(int(k) for k in meta)):
        process(a.results_dir, i, meta[str(i)], a, outdir)
    print(f"done {len(a.samples or meta)} -> {outdir}")


if __name__ == "__main__":
    main()
