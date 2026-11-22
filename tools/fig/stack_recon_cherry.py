"""Stack recon_cherry rank panels (GT/Ours rows: 5 frames | 2 recon views | BEV + traj) vertically,
(a)/(b)/(c) captions with brackets UNDER the panels, Carlito Bold (Calibri-metric).
python stack_recon_cherry.py DIR RANK [RANK ...]  -> DIR/stack_rank01_03_..{png,jpg,pdf}"""
import glob, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

D, ranks = sys.argv[1], [int(r) for r in sys.argv[2:]]
TTF = "/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf"
# panel column spans in the 2244-wide rank images (found from the white separators)
SPANS = [((48, 1351), "(a) Generation Comparison with Ground-Truth Frames"),
         ((1372, 1941), "(b) Reconstruction Results"),
         ((1962, 2243), "(c) BEV + Traj.")]
GAP, FS = 14, 50
BR_Y, BR_T = 10, 8               # bracket offset below panels, tick height
TXT_Y = BR_Y + 12                # caption top below panels
FOOT = TXT_Y + FS + 12

ims = [Image.open(glob.glob(f"{D}/rank{r:02d}_s*.jpg")[0]).convert("RGB") for r in ranks]
W = ims[0].width
PH = sum(im.height for im in ims) + GAP * (len(ims) - 1)
panels = Image.new("RGB", (W, PH), "white"); y = 0
for im in ims:
    panels.paste(im, (0, y)); y += im.height + GAP
stem = f"{D}/stack_rank" + "_".join(f"{r:02d}" for r in ranks)

# ---- raster (png/jpg)
F = ImageFont.truetype(TTF, FS)
out = Image.new("RGB", (W, PH + FOOT), "white"); out.paste(panels, (0, 0)); d = ImageDraw.Draw(out)
for (x0, x1), txt in SPANS:
    yb = PH + BR_Y
    d.line([(x0 + 4, yb), (x1 - 4, yb)], fill="black", width=3)
    d.line([(x0 + 4, yb), (x0 + 4, yb - BR_T)], fill="black", width=3)
    d.line([(x1 - 4, yb), (x1 - 4, yb - BR_T)], fill="black", width=3)
    l, t, r, b = d.textbbox((0, 0), txt, font=F)
    tx = min((x0 + x1 - (r - l)) / 2, W - 2 - (r - l))          # one line; keep inside canvas
    d.text((tx - l, PH + TXT_Y - t), txt, font=F, fill="black")
out.save(stem + ".png"); out.save(stem + ".jpg", quality=95)

# ---- pdf: raster panels + vector brackets/captions (embedded Carlito, Type 42)
matplotlib.rcParams["pdf.fonttype"] = 42
font_manager.fontManager.addfont(TTF)
fp = font_manager.FontProperties(fname=TTF)
DPI = 200; H = PH + FOOT
fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
ax.imshow(np.asarray(panels), extent=(0, W, PH, 0), interpolation="none")
lw = 3 * 72 / DPI
for (x0, x1), txt in SPANS:
    yb = PH + BR_Y
    ax.plot([x0 + 4, x0 + 4, x1 - 4, x1 - 4], [yb - BR_T, yb, yb, yb - BR_T], color="black", lw=lw,
            solid_joinstyle="miter", solid_capstyle="butt")
    tw = F.getlength(txt); tx = min((x0 + x1 - tw) / 2, W - 2 - tw)
    ax.text(tx, PH + TXT_Y, txt, fontproperties=fp, fontsize=FS * 72 / DPI,
            ha="left", va="top", color="black", wrap=False)
fig.savefig(stem + ".pdf", dpi=DPI); plt.close(fig)
print(stem, out.size)
