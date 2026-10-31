r"""
# Midterm: Local Histogram-Based Object Detection

Everything is in this one file (detector, slide-6 features, tests, CLI). Outputs go to `out/`.

```
pip install opencv-python numpy matplotlib
python histdetect.py                             # features + colorspace (default)
python histdetect.py colorspace                  # Gray vs RGB vs HSV vs Lab: stop sign vs. skin (~4 s)
python histdetect.py realtest --photo stop_photo.png --sheet sign_sheet.png   # real-image tests (~10 min) -> out/real/
python histdetect.py handtest asl_f1.png asl_f2.png asl_f3.png               # hand/skin test (~2 min) -> out/hands/
python histdetect.py features                    # slide 6: histogram features (~3 min)
python histdetect.py histograms                  # histogram figures for the slides -> out/histograms/
python histdetect.py detect scene.jpg stop.png --space rgb --metric d2 -o det.png
```

## Sections in this file (same order as the slides)
INTRODUCTION (test images) -> PART 1A color space -> PART 1B noise -> PART 2 segmentation
-> PART 3 local histograms -> PART 4 feature extraction -> PART 5 similarity matching
-> APPLICATION: hand detection -> OVERVIEW: advantages / limitations -> COMMAND LINE.

## Pipeline → spec mapping
| Spec step | Implementation |
|---|---|
| 1A Color space | `gray`, `rgb`, `hsv` (H,S; V optional via `--use-v`), `lab` (a*,b*) |
| 1B Noise reduction | Gaussian or median blur (`--blur`, `--ksize`) |
| 2 Segmentation | `roi_mask` arg: only windows centered in the mask are scanned |
| 3 Local histograms | each window split into `grid × grid` cells; per-cell, per-channel histograms via integral histograms (O(bins) per cell) |
| 4 Normalization / feature vector | each histogram ÷ pixel count, concatenated; `descriptive_stats()` gives mean, α-trimmed mean, median, mode, std |
| 5 Similarity | D1 = Σ\|a−b\|, D2 = Σ(a−b)², D3 = Σ(a−b)²/max(a,b) (the slide labels the third one D2; it's the chi-square-like one) |
| Sliding window + boxes | 24 scales (6–50% of image height), stride 10% of window, adaptive threshold (mean − 2.5σ), NMS |

Color co-occurrence histograms are intentionally not implemented.

## Slide 6: features from histograms (`histdetect.py` section 4b, `features` command)
| Feature | Function | Formula |
|---|---|---|
| Average intensity | `hist_mean` | m = Σ r_i p(r_i) |
| n-th central moment | `hist_moment` | u_n = Σ (r_i − m)^n p(r_i) |
| Variance | `hist_variance` | σ² = u_2 |
| Entropy | `hist_entropy` | e = −Σ p(r_i) log₂ p(r_i) |
| Contrast | `hist_contrast` | RMS contrast = σ |
| Area for a gray-level range | `hist_area` | MN · Σ_{lo≤r_i≤hi} p(r_i) |
| Local mean / variance | `local_mean_var` (box filters), `local_stats_from_hist` (slide formulas, reference) | m_Sxy, σ²_Sxy over neighborhood S_xy |

The `features` command checks the histogram formulas against the pixel formulas (exact match globally; ≤1e-11 for 7×7 local stats), prints the features for each region, and draws local mean/std/entropy maps (`out/local_stats.png`).

Detector option `--stats-w W` appends the key statistics per cell and channel (mean, alpha-trimmed mean, median, mode, 2σ; hue: mode only). Result on the synthetic test scene (`out/features_results.txt`): **no consistent gain**. RGB D1/D2 get worse as the weight grows (IoU 0.84 → 0.23 at w=2), HSV D2 fails at w=2, gray stays at 0, and Lab and HSV D1 move only slightly. On the street photo HSV drops 0.77 → 0.63 (`realtest`). That's expected: the statistics summarize the histogram that's already in the feature vector, so they add little new information. The local std/entropy maps are more useful as an edge/texture cue than as matching features.

## Implementation details worth mentioning in the write-up
- **Hue wrap-around**: red sits at H≈0 and H≈179; hue is shifted half a bin so red lands in one bin instead of splitting between first and last.
- **Achromatic bin**: hue is undefined for low-saturation pixels (white, gray), so pixels with S<50 go to a separate "gray" bin.
- **Scale-consistent template**: the template is resized to each window size *before* blurring/histogramming, so blur affects template and scene equally.

## Color space: stop sign vs. hands (`colorspace` command)
Step 1A's color spaces (Gray, RGB, HSV (H,S), CIE Lab (a*,b*)) plus YCrCb (Cr,Cb), the standard space for skin. Histogram backprojection, scored with pixel ROC AUC (1 = perfect, 0.5 = chance) under 5 lighting conditions. Results: `out/colorspace_results.txt`.
- Real skin (Adobe Stock portrait, `skin_photo.jpg`), normal light: Lab 0.90, HSV 0.88, YCrCb 0.86, RGB 0.83, Gray 0.66. Dimmed x0.45: HSV 0.88, Lab 0.59, YCrCb and RGB 0.50.
- Synthetic skin (10 Monk tones): HSV, Lab and YCrCb are close; Gray and RGB collapse when dimmed.
- Stop sign (sign sheet): HSV is best and the only space that holds under lighting changes.

## Real-image tests (`realtest` command)
Street photo (lenconnect.com) and a sheet of 18 road signs (Texas driving school). Results: `out/real/realtest_results.txt`, `realtest.json`, figures in `out/real/`.
- Photo: HSV without mask finds the sign with IoU 0.77 for D1/D2/D3, no false hits; RGB needs the mask (0.17 → 0.60); gray fails.
- Hue-fix ablation on the photo: removing the gray bin drops IoU to 0.14; removing the hue shift to 0.34–0.53.
- Sign sheet, sliding window: only HSV puts STOP first (D3 0.062, IoU 0.85); DO NOT ENTER is next (0.099). Gray, RGB, Lab pick DO NOT ENTER or YIELD.
- Sign sheet, ranking cropped signs: color alone ranks DO NOT ENTER above STOP; a 5 × 5 grid puts STOP first by a hair (0.228 vs 0.240).


## Hand test (`handtest` command)
Three ASL frames: two from How2Sign (Duarte et al., CVPR 2021), one from a YouTube ASL video. Skin ground truth drawn by hand in `HAND_GT`; an 8 px band around each outline is ignored. Histogram-only method: skin histogram from the signer's face -> backprojection -> Gaussian blur -> Otsu threshold. Color spaces: Gray, RGB, HSV, Lab, YCrCb.
- Skin model moved across people: AUC Gray 0.46, RGB 0.70, HSV 0.85, Lab 0.88, YCrCb 0.91.
- Masks (own-face model): YCrCb best on the How2Sign frames (0.66, 0.64); on frame 3 the beige wall falls in YCrCb's skin range (0.25) and HSV (0.54) / RGB (0.55) do better. HSV has the best mean (0.59).
- Per channel: brightness channels (R, G, B, V, L, Y) fail across people (AUC <= 0.48); Cr+Cb and hue transfer best (0.91).
- Hand-template matching with the Part 1 detector finds only the hand it came from.

## Step 4: key descriptive statistics
`cell_stats()` computes slide 4's mean, alpha-trimmed mean (alpha = 0.1), median, mode and std from each cell's normalized histogram (hue: mode only, since it is circular); `--stats-w w` appends them to the concatenated feature vector. They match the pixel-based `descriptive_stats()` (trimmed mean within 0.01). On the street photo they don't improve detection: HSV IoU 0.77 -> 0.63 at w >= 0.5.

## Step 5 database matching + advantages / limitations (`analysis` command)
`python histdetect.py analysis` -> `out/analysis/` (results, scale curve, shuffle figure).
- Database matching: query = photo stop sign, database = 18 sheet signs. Lab classifies it as STOP with every metric and grid; HSV always picks DO NOT ENTER (STOP a close second).
- Multi-scale: best match at 75 px (sign 66 px); on-sign windows 0.40-0.65 vs >= 0.89 elsewhere; threshold 0.76.
- Efficiency: 346,389 windows at 24 sizes in ~1.5 s (integral histograms).
- Illumination: HSV keeps IoU 0.77 when dimmed; brightening and color casts break every space.
- Noise: with Gaussian/median filtering IoU 0.63-0.77 up to sigma = 40; without filtering detection fails.
- Rotation: 3x3 grid holds to 45 deg (0.72), drops at 90 deg (0.48).
- Shuffled pixels: D3 = 0 with one histogram, 0.327 with the 3x3 grid.
- Step 2 segmentation (`color_roi()`): backproject the template's red (H,S) histogram, Otsu, widen 31 px, search only inside. Photo: ROI = 1.2% of the image, 4,869 instead of 346,389 windows, same box (IoU 0.77). Sign sheet: warm-colored signs cover 82%, little saved.
"""



import argparse
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT = "out"


_scene_rng = np.random.default_rng(7)


# ==============================================================================
# INTRODUCTION: test images, template and ground truth
# Test 1: street photo (PHOTO_GT). Test 2: sign sheet (SHEET_SIGNS).
# stop_sign() draws the 120 x 120 template; scene_img() is the synthetic scene used by `features`.
# ==============================================================================


def stop_sign(size):
    img = np.full((size, size, 3), 255, np.uint8)
    c, r = size / 2, size / 2 - 1
    pts = lambda rr: np.array([[c + rr * np.cos(np.pi / 8 + k * np.pi / 4),
                                c + rr * np.sin(np.pi / 8 + k * np.pi / 4)] for k in range(8)], np.int32)
    cv2.fillPoly(img, [pts(r)], (255, 255, 255))
    cv2.fillPoly(img, [pts(r * 0.9)], (30, 30, 200))
    fs = size / 110
    (tw, th), _ = cv2.getTextSize("STOP", cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, int(size / 25)))
    cv2.putText(img, "STOP", (int(c - tw / 2), int(c + th / 2)), cv2.FONT_HERSHEY_SIMPLEX,
                fs, (255, 255, 255), max(1, int(size / 25)), cv2.LINE_AA)
    mask = np.zeros((size, size), np.uint8)
    cv2.fillPoly(mask, [pts(r)], 1)
    return img, mask.astype(bool)


PHOTO_GT = (543, 145, 66, 66)


SHEET_SIGNS = {  # x, y, w, h on the 1024 x 1024 sign sheet
    "ONE WAY": (25, 25, 345, 113), "STOP": (502, 20, 297, 297), "ROAD WORK AHEAD": (10, 150, 245, 245),
    "INTERSTATE 95": (258, 168, 225, 225), "PEDESTRIAN": (757, 185, 243, 243), "WORKER (small)": (14, 405, 140, 142),
    "RAILROAD": (160, 372, 155, 155), "YIELD": (328, 417, 190, 165), "RIGHT CURVE": (530, 325, 237, 240),
    "SPEED LIMIT 55": (798, 492, 185, 230), "UNDER CONSTRUCTION": (18, 583, 200, 200), "BICYCLE": (183, 533, 175, 175),
    "DO NOT ENTER": (368, 600, 180, 180), "DEAD END": (560, 565, 220, 220), "NO U-TURN": (25, 815, 185, 185),
    "ROUTE 66": (266, 815, 188, 188), "DETOUR": (482, 823, 226, 172), "TRAFFIC LIGHT": (762, 768, 248, 250),
}


def scene_img():
    global _scene_rng
    _scene_rng = np.random.default_rng(7)
    H, W = 420, 640
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (235, 205, 170)                                       # sky
    for _ in range(40):                                            # trees
        x, y = _scene_rng.integers(0, W), _scene_rng.integers(0, 240)
        cv2.circle(img, (int(x), int(y)), int(_scene_rng.integers(25, 60)),
                   tuple(int(v) for v in (_scene_rng.integers(20, 60), _scene_rng.integers(90, 150), _scene_rng.integers(20, 60))), -1)
    cv2.fillPoly(img, [np.array([[0, 420], [640, 420], [640, 260], [0, 260]])], (70, 150, 60))   # grass
    cv2.fillPoly(img, [np.array([[150, 420], [420, 420], [330, 250], [270, 250]])], (120, 120, 125))  # road
    cv2.rectangle(img, (470, 120), (600, 250), (170, 190, 210), -1)                                 # house
    cv2.fillPoly(img, [np.array([[455, 125], [615, 125], [535, 60]])], (40, 50, 150))              # red roof (distractor)
    cv2.rectangle(img, (520, 185), (550, 250), (60, 60, 90), -1)
    cv2.rectangle(img, (60, 300), (100, 340), (35, 35, 190), -1)   # red mailbox (distractor)
    cv2.rectangle(img, (295, 300), (345, 330), (40, 40, 160), -1)  # red car (distractor)
    # stop sign on a pole
    s, sx, sy = 58, 200, 170
    cv2.rectangle(img, (sx + s // 2 - 3, sy + s), (sx + s // 2 + 3, 300), (110, 110, 110), -1)
    sign, mask = stop_sign(s)
    roi = img[sy:sy + s, sx:sx + s]
    roi[mask] = sign[mask]
    gt = (sx, sy, s, s)
    # illumination change (shadow gradient, darker overall) + noise
    shade = np.linspace(0.55, 0.95, W)[None, :, None]
    img = np.clip(img * shade + _scene_rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)
    return img, gt


# ==============================================================================
# PART 1A: COLOR SPACE CONVERSION
# Gray, RGB, HSV (H,S), Lab (a*,b*), YCrCb (Cr,Cb); brightness channel dropped by default.
# Backprojection + AUC: how well a color model isolates the target, under 5 lighting conditions.
# ==============================================================================


# per color space: (cv2 conversion, channels used, bins per channel, channel max value)
SPACES = {
    "gray": (cv2.COLOR_BGR2GRAY, [0],    [32],     [256]),
    "rgb":  (None,               [2, 1, 0], [16, 16, 16], [256, 256, 256]),
    # V dropped by default -> illumination invariance (pass --use-v to keep it)
    "hsv":  (cv2.COLOR_BGR2HSV,  [0, 1], [18, 8],  [180, 256]),
    # L dropped by default (a*, b* carry chroma)
    "lab":  (cv2.COLOR_BGR2LAB,  [1, 2], [16, 16], [256, 256]),
    "ycrcb": (cv2.COLOR_BGR2YCrCb, [1, 2], [16, 16], [256, 256]),   # Y dropped (skin/hand tests)
}


def convert_color(img, space="hsv"):
    """Step 1A: BGR -> gray / RGB / HSV / Lab / YCrCb (always returns H x W x C)."""
    conv = SPACES[space][0]
    out = cv2.cvtColor(img, conv) if conv is not None else img
    return out[..., None] if out.ndim == 2 else out


# -------------------- Backprojection and AUC (color model of the target)

CS_SPACES = {  # name: (cv2 conversion, channels, bins, ranges); step-1A options (Gray, RGB, HSV, Lab) + YCrCb for skin
    "Gray": (cv2.COLOR_BGR2GRAY, [0], [32], [0, 256]),
    "RGB": (None, [0, 1, 2], [16, 16, 16], [0, 256] * 3),
    "HSV (H,S)": (cv2.COLOR_BGR2HSV, [0, 1], [32, 32], [0, 180, 0, 256]),
    "Lab (a*,b*)": (cv2.COLOR_BGR2LAB, [1, 2], [32, 32], [0, 256, 0, 256]),
    "YCrCb (Cr,Cb)": (cv2.COLOR_BGR2YCrCb, [1, 2], [32, 32], [0, 256, 0, 256]),
}


CS_LIGHTS = {"normal": (1.0, (1, 1, 1)), "dim x0.45": (0.45, (1, 1, 1)), "bright x1.25": (1.25, (1, 1, 1)),
             "warm cast": (0.9, (0.75, 0.95, 1.15)), "cool cast": (0.95, (1.12, 1.0, 0.9))}   # gain, BGR tint


def _hex_bgr(c):
    return np.array([int(c[4:6], 16), int(c[2:4], 16), int(c[0:2], 16)], float)


def _relight(img, gain, tint):
    return np.clip(img.astype(float) * gain * np.array(tint), 0, 255).astype(np.uint8)


def _cs_bins(img, s):
    conv, ch, b, r = CS_SPACES[s]
    x = img if conv is None else cv2.cvtColor(img, conv)
    if x.ndim == 2:
        x = x[..., None]
    return tuple(np.minimum(x[..., c].astype(int) * n // r[2 * i + 1], n - 1) for i, (c, n) in enumerate(zip(ch, b)))


def _cs_model(imgs, s, masks=None):
    h = np.zeros(CS_SPACES[s][2])
    for k, img in enumerate(imgs):
        idx = _cs_bins(img, s)
        if masks is not None:
            idx = tuple(a[masks[k]] for a in idx)
        np.add.at(h, idx, 1)
    return h / h.sum()


def _cs_backproject(img, h, s):
    return h[_cs_bins(img, s)]


def _auc(pos, neg):
    """ROC AUC via Mann-Whitney U (ties handled by average ranks)."""
    from scipy.stats import rankdata
    r = rankdata(np.r_[pos, neg])
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


# -------------------- Figure: stop-sign colors as RGB and HSV histograms

def fig_color_hists(photo, sheet, tmpl, path):
    """Template vs the two real STOP signs as RGB and HSV histograms (whole sign, one cell)."""
    pb, sb = PHOTO_GT, SHEET_SIGNS["STOP"]
    items = [("Template", tmpl, None, "#7f7f7f"), ("Photo sign", photo, pb, "#c0392b"), ("Sheet STOP", sheet, sb, "#2e86c1")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.2), gridspec_kw={"width_ratios": [48, 27]})
    res = {}
    for ax, sp, groups in [(axes[0], "rgb", [("R (16 bins)", 16), ("G (16 bins)", 16), ("B (16 bins)", 16)]),
                           (axes[1], "hsv", HSV_GROUPS)]:
        hs = []
        for name, img, box, _ in items:
            if box is None:
                t = cv2.resize(img, (sb[2], sb[3]), interpolation=cv2.INTER_AREA); hs.append(region_hists(t, None, sp))
            else:
                hs.append(region_hists(img, box, sp))
        dp, ds = hist_dist(hs[0], hs[1]), hist_dist(hs[0], hs[2])
        res[sp] = (dp, ds)
        _bars(ax, hs, [f"{n}" + ("" if i == 0 else f" (D3 to template {[dp, ds][i - 1]:.2f})") for i, (n, *_r) in enumerate(items)],
              [c for *_r, c in items], groups, title=f"{sp.upper()}" if sp == "rgb" else "HSV (H, S)", ylim=0.95)
        ax.legend(fontsize=7.5, loc="upper left" if sp == "rgb" else "upper center", frameon=False)
        ax.set_ylabel("share of pixels", fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    return res


# ==============================================================================
# PART 1B: NOISE REDUCTION
# Gaussian (default) or median 5 x 5, applied before every histogram.
# ==============================================================================


def reduce_noise(img, blur="gaussian", ksize=5):
    """Step 1B: Gaussian or median filter (k x k)."""
    if blur == "gaussian":
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    if blur == "median":
        return cv2.medianBlur(img, ksize)
    return img


def preprocess(img, space="hsv", blur="gaussian", ksize=5):
    """Step 1 = 1B then 1A: filter noise, then convert the color space."""
    return convert_color(reduce_noise(img, blur, ksize), space)


# -------------------- Figure: a noisy cell histogram, with and without filtering

def fig_noise_hists(photo, tmpl, path, sigma=20, seed=0):
    """Center cell of the sign (3 x 3 grid), HSV, with camera-like noise: no filter vs Gaussian vs median."""
    x, y, w, h = PHOTO_GT
    noisy = np.clip(photo.astype(float) + np.random.default_rng(seed).normal(0, sigma, photo.shape), 0, 255).astype(np.uint8)
    t = cv2.resize(tmpl, (w, h), interpolation=cv2.INTER_AREA)
    cell = lambda hs: hs[8:10]                       # center cell = 5th of 9 cells, 2 histograms each
    rows = [("Template", cell(region_hists(t, None, "hsv", grid=3)), "#7f7f7f")]
    res = {}
    for name, img, blur in [(f"Noise σ={sigma}, no filter", noisy, "none"),
                            (f"Noise σ={sigma}, Gaussian 5×5", noisy, "gaussian"), (f"Noise σ={sigma}, median 5×5", noisy, "median")]:
        hc = cell(region_hists(img, PHOTO_GT, "hsv", grid=3, blur=blur))
        res[name] = hist_dist(hc, cell(region_hists(t, None, "hsv", grid=3, blur=blur)))   # same filter on both
        rows.append((f"{name.split(', ')[1]}: D3 {res[name]:.2f}", hc, "#c0392b" if blur == "none" else "#27ae60"))
    fig, axes = plt.subplots(1, len(rows), figsize=(10, 2.4), sharey=True)
    for ax, (name, hs, c) in zip(axes, rows):
        _bars(ax, [hs], [name], [c], [("hue", 18), ("g", 1), ("sat", 8)], title=name, ylim=1.0)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    return res


# ==============================================================================
# PART 2: SEGMENTATION
# Color region of interest from the template's red, then the sliding window only inside it.
# detect(..., roi_mask=...) in Part 5 does the restricted search.
# ==============================================================================


def color_roi(img, template, dilate=31):
    """Step 2 (segmentation): regions whose colors match the template's colored pixels.
    HSV (H,S) histogram of the template's saturated pixels -> backprojection -> blur + Otsu -> dilate."""
    tb = cv2.cvtColor(cv2.GaussianBlur(template, (5, 5), 0), cv2.COLOR_BGR2HSV)
    h = _cs_model([template], "HSV (H,S)", [tb[..., 1] >= SAT_MIN])
    bp = _cs_backproject(cv2.GaussianBlur(img, (5, 5), 0), h, "HSV (H,S)")
    m = _skin_mask(bp)
    return cv2.dilate(m.astype(np.uint8), np.ones((dilate, dilate), np.uint8)).astype(bool)


def run_segmentation(photo, sheet, tmpl, od, log, R):
    import time
    log("\n8) STEP 2 SEGMENTATION: color ROI from the template's colors, then the sliding window only inside it (HSV, D3)")
    R["segmentation"] = {}
    for name, img, gt in [("photo", photo, PHOTO_GT), ("sheet", sheet, SHEET_SIGNS["STOP"])]:
        for dil in [1, 31]:
            m = color_roi(img, tmpl, dil)
            for use in [False, True]:
                t0 = time.perf_counter()
                dets, maps, thr = detect(img, tmpl, space="hsv", metric="d3", roi_mask=m if use else None)
                dt = time.perf_counter() - t0
                d, b = _best_window(maps)
                r = dict(roi_area=float(m.mean()), windows=sum(len(bs) for bs, _ in maps.values()), seconds=dt,
                         best=b, iou=iou(b, gt), threshold=float(thr), n_det=len(dets),
                         det_on_target=sum(1 for _, bb in dets if iou(bb, gt) > 0))
                R["segmentation"][f"{name}|dilate{dil}|roi={use}"] = r
                log(f"   {name:5} dilate {dil:2d} ROI={str(use):5}: area {r['roi_area']:.3f}  windows {r['windows']:7d}  "
                    f"time {dt:.2f} s  best IoU {r['iou']:.2f}  detections {r['n_det']} ({r['det_on_target']} on target)")
    m = color_roi(photo, tmpl, 31)
    vis = photo.copy(); vis[~m] = (vis[~m] * 0.25).astype(np.uint8)
    x, y, w, h = PHOTO_GT; cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 255), 2)
    fig, ax = plt.subplots(1, 2, figsize=(10, 2.9))
    ax[0].imshow(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)); ax[0].set_title("Input", fontsize=10); ax[0].axis("off")
    ax[1].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); ax[1].set_title(f"Region of interest ({100 * m.mean():.1f}% of the image)", fontsize=10); ax[1].axis("off")
    plt.tight_layout(); plt.savefig(f"{od}/roi_photo.png", dpi=150); plt.close()


# ==============================================================================
# PART 3: LOCAL HISTOGRAM CALCULATION
# Window -> grid x grid cells -> one histogram per cell per channel, via integral histograms.
# Hue fixes: achromatic gray bin (S < SAT_MIN) and the hue shift so red stays in one bin.
# ==============================================================================


# -------------------- Hue fixes

SAT_MIN = 50   # HSV pixels below this saturation go to the achromatic hue bin


def achromatic(img, space):
    return img[..., 1] < SAT_MIN if space == "hsv" else None


def circular_channels(space):
    return (0,) if space == "hsv" else ()


def channel_spec(space, use_v=False):
    _, ch, bins, maxv = SPACES[space]
    ch, bins, maxv = list(ch), list(bins), list(maxv)
    if use_v and space in ("hsv", "lab"):
        extra = 2 if space == "hsv" else 0
        ch.append(extra); bins.append(8); maxv.append(256)
    return ch, bins, maxv


# -------------------- Integral histogram and window features

class IntegralHistogram:
    """One integral image per (channel, bin) -> any rectangle's histogram in O(bins).
    circular: channel indices that wrap (HSV hue); mask: optional per-pixel weights."""

    def __init__(self, img, channels, bins, maxv, circular=(), mask=None, achrom=None):
        planes, sizes, self.centers = [], [], []
        wgt = np.ones(img.shape[:2], np.float32) if mask is None else mask.astype(np.float32)
        for c, b, m in zip(channels, bins, maxv):
            v = img[..., c].astype(np.int32)
            if c in circular:                             # center bin 0 on hue 0 (red wraps 179->0)
                v = (v + m // (2 * b)) % m
            idx = np.minimum(v * b // m, b - 1)
            if c in circular and achrom is not None:      # extra bin for gray/white pixels (hue undefined)
                idx = np.where(achrom, b, idx); b += 1
            sizes.append(b)
            self.centers.append(None if c in circular else (np.arange(b) + 0.5) / b)  # r_i in [0,1]
            onehot = (idx[..., None] == np.arange(b)).astype(np.float32) * wgt[..., None]
            ii = np.zeros((img.shape[0] + 1, img.shape[1] + 1, b), np.float32)
            ii[1:, 1:] = onehot.cumsum(0).cumsum(1)
            planes.append(ii)
        self.ii = np.concatenate(planes, axis=2)          # H+1 x W+1 x total_bins
        self.splits = np.cumsum(sizes)[:-1]

    def rect(self, y0, x0, y1, x1):
        ii = self.ii
        return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]


def window_feature(ih, y, x, h, w, grid, stats_w=0.0):
    """Grid of cells -> per-cell, per-channel histograms, each normalized to sum 1."""
    return batch_features(ih, [(x, y, w, h)], grid, stats_w)[0]


def batch_features(ih, boxes, grid, stats_w=0.0):
    """Feature vectors for many (x, y, w, h) boxes.
    stats_w > 0 appends slide-4 statistics (mean, alpha-trimmed mean, median, mode, std) per cell & channel, scaled by stats_w."""
    b = np.asarray(boxes)
    t = np.linspace(0, 1, grid + 1)
    ys = (b[:, 1, None] + t * b[:, 3, None]).astype(int)          # N x g+1
    xs = (b[:, 0, None] + t * b[:, 2, None]).astype(int)
    Y, X = ys[:, :, None], xs[:, None, :]                          # N x g+1 x g+1
    c = ih.ii[Y, X]                                                # N x g+1 x g+1 x B
    cells = c[:, 1:, 1:] - c[:, :-1, 1:] - c[:, 1:, :-1] + c[:, :-1, :-1]
    parts = np.split(cells, ih.splits, axis=-1)
    parts = [p / np.maximum(p.sum(-1, keepdims=True), 1e-9) for p in parts]   # p(r_i)
    feats = parts
    if stats_w > 0:
        feats = feats + [stats_w * cell_stats(p, r) for p, r in zip(parts, ih.centers)]
    return np.concatenate(feats, -1).reshape(len(b), -1)


# -------------------- Histogram of one region, and the hue-fix figure

def region_hists(img, box=None, space="hsv", grid=1, blur="gaussian", ksize=5, hue_shift=True, gray_bin=True):
    """Normalized per-cell, per-channel histograms of one box, built as the detector builds them.
    hue_shift / gray_bin = False switch off the hue fixes (for the Part 3 figure)."""
    x, y, w, h = box if box is not None else (0, 0, img.shape[1], img.shape[0])
    ch, bins, maxv = channel_spec(space)
    p = preprocess(img[y:y + h, x:x + w], space, blur, ksize)
    ach = achromatic(p, space) if gray_bin else None
    circ = circular_channels(space) if hue_shift else ()
    ih = IntegralHistogram(p, ch, bins, maxv, circ, achrom=ach)
    out = []
    for gy in range(grid):
        for gx in range(grid):
            v = ih.rect(gy * h // grid, gx * w // grid, (gy + 1) * h // grid, (gx + 1) * w // grid)
            out += [q / max(q.sum(), 1e-12) for q in np.split(v, ih.splits)]
    return out


def hist_dist(a, b, metric="d3"):
    """Detector distance between two lists of histograms (divided by the number of histograms)."""
    return float(METRICS[metric](np.concatenate(a), np.concatenate(b)) / len(a))


def _bars(ax, hs, labels, colors, groups, title=None, ylim=None):
    """Grouped bars of concatenated histograms; groups = [(name, n_bins), ...] marked by separators."""
    n = sum(g[1] for g in groups); x = np.arange(n); k = len(hs); bw = 0.8 / k
    for i, (h, lab, c) in enumerate(zip(hs, labels, colors)):
        ax.bar(x - 0.4 + bw * (i + 0.5), np.concatenate(h), bw, color=c, label=lab, linewidth=0)
    pos = 0
    for name, nb in groups:
        if pos:
            ax.axvline(pos - 0.5, color="0.6", lw=0.8)
        ax.text(pos + nb / 2 - 0.5, 1.01, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8)
        pos += nb
    ax.set_xlim(-0.6, n - 0.4); ax.set_xticks([])
    if ylim:
        ax.set_ylim(0, ylim)
    ax.tick_params(labelsize=7)
    if title:
        ax.set_title(title, fontsize=9.5, pad=13)


HSV_GROUPS = [("hue (18 bins)", 18), ("gray", 1), ("saturation (8 bins)", 8)]


def fig_hue_fixes(photo, path):
    """Hue histogram of the photo's stop sign with and without the two hue fixes."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.5), sharey=True)
    cases = [("No fixes: red split in bins 0 and 17,\nwhite letters get random hues", False, False),
             ("Hue shift: most of the red moves to bin 0", True, False),
             ("Shift + gray bin: white goes to 'gray'", True, True)]
    for ax, (title, sh, gb) in zip(axes, cases):
        hue = region_hists(photo, PHOTO_GT, "hsv", hue_shift=sh, gray_bin=gb)[0]
        cols = ["#c0392b" if (i in (0, 17)) else "#7f8c8d" for i in range(18)] + (["#bdc3c7"] if gb else [])
        ax.bar(np.arange(len(hue)), hue, color=cols, linewidth=0)
        ax.set_xticks([0, 17] + ([18] if gb else [])); ax.set_xticklabels(["0", "17"] + (["gray"] if gb else []), fontsize=7)
        ax.set_title(title, fontsize=9); ax.tick_params(labelsize=7)
    axes[0].set_ylabel("share of pixels", fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


# ==============================================================================
# PART 4: FEATURE EXTRACTION
# Normalized histograms concatenated into the feature vector (batch_features, Part 3).
# Key statistics (mean, alpha-trimmed mean, median, mode, std) and the slide 6 features.
# ==============================================================================


# -------------------- Slide 6 features from a histogram

# All computed from a normalized histogram p(r_i), i = 0..L-1, with gray levels r_i.
def hist_mean(p, r):
    """m = sum r_i p(r_i)"""
    return (p * r).sum(-1)


def hist_moment(p, r, n):
    """u_n = sum (r_i - m)^n p(r_i)"""
    m = hist_mean(p, r)
    return (((r - m[..., None]) ** n) * p).sum(-1)


def hist_variance(p, r):
    """sigma^2 = u_2"""
    return hist_moment(p, r, 2)


def hist_entropy(p):
    """e = -sum p log2 p (bits)"""
    q = np.where(p > 0, p, 1)
    return -(p * np.log2(q)).sum(-1)


def hist_contrast(p, r):
    """RMS contrast = sigma (std of intensity); used as an image-quality measure."""
    return np.sqrt(np.maximum(hist_variance(p, r), 0))


def hist_area(p, r, lo, hi, n_pixels=1):
    """Area for gray-level range [lo, hi]: pixel count (or fraction if n_pixels=1)."""
    return n_pixels * (p * ((r >= lo) & (r <= hi))).sum(-1)


def hist_stats(hist, levels=None):
    """All slide-6 features from a raw or normalized 1-D histogram."""
    hist = np.asarray(hist, np.float64)
    n = hist.sum()
    p = hist / n
    r = np.arange(len(p)) if levels is None else np.asarray(levels)
    return dict(mean=hist_mean(p, r), variance=hist_variance(p, r), std=hist_contrast(p, r),
                u3=hist_moment(p, r, 3), entropy=hist_entropy(p), contrast=hist_contrast(p, r),
                n_pixels=n)


# -------------------- Key descriptive statistics

def hist_median(p, r):
    """Median: first level where the CDF reaches 0.5."""
    cdf = np.cumsum(p, -1)
    return np.take(r, np.argmax(cdf >= 0.5 - 1e-12, axis=-1))


def hist_mode(p, r):
    """Mode: level of the highest bin."""
    return np.take(r, np.argmax(p, axis=-1))


def hist_trimmed_mean(p, r, alpha=0.1):
    """Alpha-trimmed mean: drop the lowest and highest alpha of the mass, average the rest."""
    cdf = np.cumsum(p, -1)
    lo = np.concatenate([np.zeros_like(cdf[..., :1]), cdf[..., :-1]], -1)
    w = np.clip(np.minimum(cdf, 1 - alpha) - np.maximum(lo, alpha), 0, None)
    return (w * r).sum(-1) / np.maximum(w.sum(-1), 1e-12)


def cell_stats(p, r):
    """Slide-4 key descriptive statistics of one cell histogram, levels scaled to [0, 1]:
    [mean, alpha-trimmed mean, median, mode, 2*std]. Hue (r=None, circular) -> mode only."""
    if r is None:
        b = p.shape[-1]
        return (np.argmax(p, axis=-1) / (b - 1))[..., None]
    return np.stack([hist_mean(p, r), hist_trimmed_mean(p, r), hist_median(p, r), hist_mode(p, r),
                     2 * hist_contrast(p, r)], -1)


def descriptive_stats(patch, alpha=0.1):
    """Mean, alpha-trimmed mean, median, mode, std per channel."""
    out = {}
    for c in range(patch.shape[2]):
        v = np.sort(patch[..., c].ravel().astype(np.float64))
        k = int(alpha * len(v))
        trimmed = v[k:len(v) - k] if len(v) > 2 * k else v
        out[c] = dict(mean=v.mean(), alpha_trimmed=trimmed.mean(), median=np.median(v),
                      mode=np.bincount(v.astype(int)).argmax(), std=v.std())
    return out


# -------------------- Local mean and variance

def local_mean_var(gray, k=7):
    """Local average intensity m_Sxy and local variance sigma^2_Sxy over a k x k neighborhood S_xy.
    Box filters compute the same quantity as the histogram formulas, in O(1) per pixel."""
    g = gray.astype(np.float64)
    m = cv2.blur(g, (k, k), borderType=cv2.BORDER_REFLECT)
    m2 = cv2.blur(g * g, (k, k), borderType=cv2.BORDER_REFLECT)
    return m, np.maximum(m2 - m * m, 0)


def local_stats_from_hist(gray, y, x, k=7, levels=256):
    """Reference version: m_Sxy, sigma^2_Sxy from the neighborhood histogram (slide formulas)."""
    hk = k // 2
    patch = gray[max(0, y - hk):y + hk + 1, max(0, x - hk):x + hk + 1]
    p = np.bincount(patch.ravel(), minlength=levels) / patch.size
    r = np.arange(levels)
    return hist_mean(p, r), hist_variance(p, r)


# -------------------- `features` command

def run_features():
    _feat_rng = np.random.default_rng(0)
    os.makedirs(OUT, exist_ok=True)
    scene, gt = scene_img()
    tmpl, tmask = stop_sign(120)
    gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    lines = []
    log = lambda t="": (print(t), lines.append(t))

    # 1. histogram formulas == pixel formulas
    hist = np.bincount(gray.ravel(), minlength=256)
    st = hist_stats(hist)
    MN = gray.size
    m_pix = gray.sum() / MN                                    # (1/MN) sum f(x,y)
    v_pix = ((gray.astype(np.float64) - m_pix) ** 2).sum() / MN
    log("1) Global: histogram formula vs pixel formula")
    log(f"   mean      {st['mean']:.6f}  vs  {m_pix:.6f}")
    log(f"   variance  {st['variance']:.6f}  vs  {v_pix:.6f}")

    m_map, v_map = local_mean_var(gray, 7)
    errs = []
    for _ in range(200):
        y, x = _feat_rng.integers(3, gray.shape[0] - 3), _feat_rng.integers(3, gray.shape[1] - 3)
        mh, vh = local_stats_from_hist(gray, y, x, 7)
        errs.append(max(abs(mh - m_map[y, x]), abs(vh - v_map[y, x])))
    log(f"   local (7x7, 200 random pixels): max |box filter - histogram formula| = {max(errs):.2e}")

    # 2. slide-6 features for a few regions (grayscale, L = 256)
    regions = {"whole scene": (0, 0, gray.shape[1], gray.shape[0]), "stop sign": gt,
               "road": (300, 340, 58, 58), "tree": (100, 60, 58, 58), "sky": (360, 10, 58, 40)}
    log("\n2) Histogram features per region (gray levels 0-255)")
    log(f"   {'region':12} {'mean':>7} {'var':>9} {'contrast':>9} {'u3':>11} {'entropy':>8} {'area[0,63]':>11} {'area[192,255]':>13}")
    for name, (x, y, w, h) in regions.items():
        h_ = np.bincount(gray[y:y + h, x:x + w].ravel(), minlength=256)
        s = hist_stats(h_)
        p, r = h_ / h_.sum(), np.arange(256)
        dark, bright = hist_area(p, r, 0, 63, h_.sum()), hist_area(p, r, 192, 255, h_.sum())
        log(f"   {name:12} {s['mean']:7.1f} {s['variance']:9.1f} {s['contrast']:9.1f} {s['u3']:11.0f} "
            f"{s['entropy']:8.2f} {dark:11.0f} {bright:13.0f}")

    # 3. local feature maps
    k = 9
    m_map, v_map = local_mean_var(gray, k)
    ih = IntegralHistogram(gray[..., None], [0], [32], [256])
    H, W = gray.shape
    yy, xx = np.mgrid[0:H, 0:W]
    y0, y1 = np.clip(yy - k // 2, 0, H), np.clip(yy + k // 2 + 1, 0, H)
    x0, x1 = np.clip(xx - k // 2, 0, W), np.clip(xx + k // 2 + 1, 0, W)
    hloc = ih.ii[y1, x1] - ih.ii[y0, x1] - ih.ii[y1, x0] + ih.ii[y0, x0]
    ent_map = hist_entropy(hloc / hloc.sum(-1, keepdims=True))
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
    for a, img, t, cm in zip(ax, [gray, m_map, np.sqrt(v_map), ent_map],
                             ["Grayscale f(x,y)", f"Local mean $m_{{S_{{xy}}}}$ ({k}×{k})",
                              f"Local std $\\sigma_{{S_{{xy}}}}$ (contrast)", "Local entropy (32 bins, bits)"],
                             ["gray", "gray", "magma", "viridis"]):
        im = a.imshow(img, cmap=cm); a.set_title(t); a.axis("off"); plt.colorbar(im, ax=a, fraction=0.03)
    plt.tight_layout(); plt.savefig(f"{OUT}/local_stats.png", dpi=110); plt.close()

    # 4. do the statistics help detection?
    log("\n4) Detection with the key statistics appended to each cell (masked template)")
    log(f"   {'space':6} {'metric':6} " + " ".join(f"{'w=' + str(w):>12}" for w in [0, 0.5, 1, 2]))
    for space in ["gray", "rgb", "hsv", "lab"]:
        for m in ["d1", "d2", "d3"]:
            cells = []
            for w in [0, 0.5, 1, 2]:
                dets, _, _ = detect(scene, tmpl, space=space, metric=m, template_mask=tmask, stats_w=w)
                i = iou(dets[0][1], gt) if dets else 0
                cells.append(f"{i:.2f}{'*' if i >= 0.5 else ' '}")
            log(f"   {space:6} {m:6} " + " ".join(f"{c:>12}" for c in cells))
    log("   (values = IoU of top-1 detection with ground truth; * = IoU >= 0.5)")
    open(f"{OUT}/features_results.txt", "w").write("\n".join(lines) + "\n")


# ==============================================================================
# PART 5: SIMILARITY MATCHING AND CLASSIFICATION
# D1, D2, D3; multi-scale sliding window; adaptive threshold; NMS; bounding boxes.
# `realtest` runs Tests 1 and 2; `detect` runs on your own images.
# ==============================================================================


# -------------------- Distances

def d1(a, b):  # sum |a-b|
    return np.abs(a - b).sum(-1)


def d2(a, b):  # sum (a-b)^2
    return ((a - b) ** 2).sum(-1)


def d3(a, b):  # sum (a-b)^2 / max(a,b)   (chi-square-like)
    return ((a - b) ** 2 / np.maximum(np.maximum(a, b), 1e-9)).sum(-1)


METRICS = {"d1": d1, "d2": d2, "d3": d3}


# -------------------- Sliding window, threshold, NMS

def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter)


def nms(dets, thr=0.3):
    """dets: list of (dist, box); lower distance = better."""
    keep = []
    for d in sorted(dets, key=lambda t: t[0]):
        if all(iou(d[1], k[1]) < thr for k in keep):
            keep.append(d)
    return keep


def detect(scene, template, space="hsv", metric="d3", grid=3, scales=None,
           stride=0.1, thresh=None, blur="gaussian", ksize=5, use_v=False,
           roi_mask=None, template_mask=None, top_k=5, stats_w=0.0):
    """Returns (detections, score_maps, threshold). Detection = (distance, (x, y, w, h)).
    template_mask: nonzero = object pixels (e.g. PNG alpha); background is ignored."""
    ch, bins, maxv = channel_spec(space, use_v)
    s_img = preprocess(scene, space, blur, ksize)

    circ = circular_channels(space)
    th, tw = template.shape[:2]

    def target_at(h, w):
        """Template resized to window size, then preprocessed -> scale-consistent blur/detail."""
        t = cv2.resize(template, (w, h), interpolation=cv2.INTER_AREA)
        m = None if template_mask is None else cv2.resize(
            template_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        tp = preprocess(t, space, blur, ksize)
        ih = IntegralHistogram(tp, ch, bins, maxv, circ, m, achromatic(tp, space))
        return window_feature(ih, 0, 0, h, w, grid, stats_w)
    n_hist = grid * grid * len(ch)                      # rescale distance to per-histogram

    s_ih = IntegralHistogram(s_img, ch, bins, maxv, circ, achrom=achromatic(s_img, space))
    H, W = scene.shape[:2]
    if scales is None:                                  # window heights as a fraction of scene height
        scales = np.geomspace(0.06, 0.5, 24)
    dist_fn = METRICS[metric]

    all_dets, maps = [], {}
    for s in scales:
        h = int(round(s * H)); w = int(round(h * tw / th))
        if h < grid * 2 or w < grid * 2 or h > H or w > W:
            continue
        step = max(1, int(stride * min(h, w)))
        boxes = [(x, y, w, h) for y in range(0, H - h + 1, step)
                 for x in range(0, W - w + 1, step)]
        if roi_mask is not None:                        # 2. segmentation (ROI mask)
            boxes = [b for b in boxes if roi_mask[b[1] + h // 2, b[0] + w // 2]]
        if not boxes:
            continue
        dists = dist_fn(batch_features(s_ih, boxes, grid, stats_w), target_at(h, w)) / n_hist
        maps[h] = (boxes, dists)
        all_dets += list(zip(dists.tolist(), boxes))

    if thresh is None:                                  # adaptive: mean - 2.5 std over all windows
        d = np.array([t[0] for t in all_dets])
        thresh = d.mean() - 2.5 * d.std()
    hits = [t for t in all_dets if t[0] <= thresh]
    return nms(hits)[:top_k], maps, thresh


def draw(scene, dets, color=(0, 255, 0)):
    out = scene.copy()
    for rank, (dist, (x, y, w, h)) in enumerate(dets):
        c = color if rank == 0 else (0, 200, 255)
        cv2.rectangle(out, (x, y), (x + w, y + h), c, 2)
        label = f"#1 d={dist:.3f}" if rank == 0 else f"#{rank+1}"
        cv2.putText(out, label, (x, max(12, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    return out


# -------------------- Tests 1 and 2 (`realtest` command)

def _best_window(maps):
    """Lowest-distance window over all scales (independent of the threshold)."""
    return min(((float(d), tuple(int(v) for v in b)) for _, (bs, ds) in maps.items() for b, d in zip(bs, ds)),
               key=lambda t: t[0])


def _eval(scene, tmpl, gt, **kw):
    dets, maps, thr = detect(scene, tmpl, top_k=100, **kw)
    d, b = _best_window(maps)
    return dict(iou=iou(b, gt), d=d, box=b, passed=bool(d <= thr),
                fp=sum(1 for _, bb in dets if iou(bb, gt) == 0), dets=dets, maps=maps)


def _sign_features(img, box, tmpl, space, grid, use_mask, tmask):
    """Feature of a sign box and of the template resized to that box (same pipeline as detect())."""
    ch, bins, maxv = channel_spec(space)
    circ = circular_channels(space)
    x, y, w, h = box
    sp = preprocess(img, space)
    s_ih = IntegralHistogram(sp, ch, bins, maxv, circ, achrom=achromatic(sp, space))
    t = cv2.resize(tmpl, (w, h), interpolation=cv2.INTER_AREA)
    m = cv2.resize(tmask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) if use_mask else None
    tp = preprocess(t, space)
    t_ih = IntegralHistogram(tp, ch, bins, maxv, circ, m, achromatic(tp, space))
    return window_feature(s_ih, y, x, h, w, grid), window_feature(t_ih, 0, 0, h, w, grid), grid * grid * len(ch)


def run_realtest(photo_path, sheet_path):
    import json
    od = os.path.join(OUT, "real"); os.makedirs(od, exist_ok=True)
    photo, sheet = cv2.imread(photo_path), cv2.imread(sheet_path)
    if photo is None or sheet is None:
        raise SystemExit(f"could not read {photo_path} or {sheet_path}")
    tmpl, tmask = stop_sign(120)
    R, lines = {}, []
    log = lambda t="": (print(t), lines.append(t))
    spaces, metrics = ["gray", "rgb", "hsv", "lab"], ["d1", "d2", "d3"]

    # ---- A. photo: color space x metric x mask
    log(f"A) PHOTO {photo_path}  ground truth {PHOTO_GT}")
    log(f"   {'mask':5} {'space':5} " + "  ".join(f"{m:>14}" for m in metrics) + "   (IoU, *=passed threshold, false hits)")
    R["photo_grid"] = {}
    for use_mask in [False]:                                   # template mask removed (outside midterm scope)
        for sp in spaces:
            cells = []
            for m in metrics:
                r = _eval(photo, tmpl, PHOTO_GT, space=sp, metric=m, template_mask=tmask if use_mask else None)
                R["photo_grid"][f"{int(use_mask)}-{sp}-{m}"] = {k: r[k] for k in ("iou", "d", "box", "passed", "fp")}
                cells.append(f"{r['iou']:.2f}{'*' if r['passed'] else ' '} fp={r['fp']:<3}")
            log(f"   {str(use_mask):5} {sp:5} " + "  ".join(f"{c:>14}" for c in cells))

    # ---- B. photo: ablation of the two hue fixes (HSV, no mask)
    global SAT_MIN
    log("\nB) PHOTO ablation, HSV, no mask")
    R["ablation"] = {}

    def abl(label):
        out = {}
        for m in ["d1", "d3"]:
            r = _eval(photo, tmpl, PHOTO_GT, space="hsv", metric=m)
            out[m] = {k: r[k] for k in ("iou", "box", "fp")}
        R["ablation"][label] = out
        log(f"   {label:26} " + "  ".join(f"{m}: IoU {v['iou']:.2f} box {v['box']} fp {v['fp']}" for m, v in out.items()))
    abl("both fixes")
    SAT_MIN = 0; abl("no gray bin"); SAT_MIN = 50
    g = globals(); orig_circ = g["circular_channels"]
    g["circular_channels"] = lambda s: (); abl("no hue shift, no gray bin"); g["circular_channels"] = orig_circ
    orig_init = IntegralHistogram.__init__

    def init_noshift(self, img, channels, bins, maxv, circular=(), mask=None, achrom=None):
        orig_init(self, img, channels, bins, maxv, (), mask, None)     # plain hue bins...
        if achrom is not None and 0 in circular:                       # ...plus the gray bin, no shift
            b0 = bins[0]
            v = img[..., 0].astype(np.int32)
            idx = np.where(achrom, b0, np.minimum(v * b0 // maxv[0], b0 - 1))
            wgt = np.ones(img.shape[:2], np.float32) if mask is None else mask.astype(np.float32)
            oh = (idx[..., None] == np.arange(b0 + 1)).astype(np.float32) * wgt[..., None]
            ii = np.zeros((img.shape[0] + 1, img.shape[1] + 1, b0 + 1), np.float32)
            ii[1:, 1:] = oh.cumsum(0).cumsum(1)
            self.ii = np.concatenate([ii, self.ii[..., b0:]], axis=2)
            self.splits = self.splits + 1
            self.centers = [None] + self.centers[1:]
    IntegralHistogram.__init__ = init_noshift; abl("no hue shift (gray bin kept)"); IntegralHistogram.__init__ = orig_init

    # ---- photo figures
    x, y, w, h = PHOTO_GT
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw=dict(width_ratios=[3.2, 1.1]))
    vis = photo.copy(); cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 255), 2)
    ax[0].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); ax[0].set_title(f"Scene ({photo.shape[1]} × {photo.shape[0]}), ground truth in magenta", fontsize=10); ax[0].axis("off")
    ax[1].imshow(cv2.cvtColor(tmpl, cv2.COLOR_BGR2RGB)); ax[1].set_title("Template (120 × 120), drawn in code", fontsize=10); ax[1].axis("off")
    plt.tight_layout(); plt.savefig(f"{od}/photo_scene_template.png", dpi=160); plt.close()

    ch, bins, maxv = channel_spec("hsv")
    pp = preprocess(photo); tp = preprocess(cv2.resize(tmpl, (w, h), interpolation=cv2.INTER_AREA))
    s_ih = IntegralHistogram(pp, ch, bins, maxv, (0,), achrom=achromatic(pp, "hsv"))
    t_ih = IntegralHistogram(tp, ch, bins, maxv, (0,), achrom=achromatic(tp, "hsv"))
    tf = window_feature(t_ih, 0, 0, h, w, 1)
    blocks = {"stop sign": PHOTO_GT, "red-leaf tree": (300, 30, 66, 66), "street sign": (540, 72, 66, 50),
              "house siding": (330, 160, 66, 50), "grass": (400, 225, 66, 60), "road": (150, 320, 66, 66)}
    fig, axes = plt.subplots(2, len(blocks) + 1, figsize=(16, 5))
    for k, (name, box) in enumerate(list(blocks.items()) + [("TARGET", None)]):
        if box is None:
            patch, f = tmpl, tf
        else:
            bx, by, bw, bh = box; patch = photo[by:by + bh, bx:bx + bw]; f = window_feature(s_ih, by, bx, bh, bw, 1)
        axes[0, k].imshow(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)); axes[0, k].axis("off")
        axes[0, k].set_title(name if box is None else f"{name}\nD1={d1(f, tf)/2:.2f}  D3={d3(f, tf)/2:.2f}", fontsize=10)
        axes[1, k].bar(range(len(f)), f, color=["#c0392b"] * bins[0] + ["#999999"] + ["#2c3e50"] * bins[1])
        axes[1, k].set_xticks([bins[0] / 2, bins[0], bins[0] + 1 + bins[1] / 2], ["Hue", "gray", "Sat"]); axes[1, k].set_yticks([])
    plt.suptitle("Local HSV histograms vs. target (lower distance = more similar)")
    plt.tight_layout(); plt.savefig(f"{od}/photo_hist_compare.png", dpi=110); plt.close()

    def heat(img, gt, fname, title_det, n_show=1):
        dets, maps, thr = detect(img, tmpl, space="hsv", metric="d3")
        vis = draw(img, dets[:n_show]); gx, gy, gw, gh = gt
        cv2.rectangle(vis, (gx, gy), (gx + gw, gy + gh), (255, 0, 255), 1 if img is photo else 3)
        bh_ = dets[0][1][3]; boxes, dists = maps[bh_]
        xs = sorted({b[0] for b in boxes}); ys = sorted({b[1] for b in boxes})
        grid = np.full((len(ys), len(xs)), np.nan)
        for b, dd in zip(boxes, dists):
            grid[ys.index(b[1]), xs.index(b[0])] = -dd
        half, step = bh_ / 2, (xs[1] - xs[0]) if len(xs) > 1 else 1
        fs = (13, 3.9) if img is photo else (11, 5.2)
        fig, ax = plt.subplots(1, 2, figsize=fs)
        ax[0].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); ax[0].set_title(title_det); ax[0].axis("off")
        im = ax[1].imshow(grid, cmap="magma", interpolation="nearest",
                          extent=(xs[0] + half - step / 2, xs[-1] + half + step / 2, ys[-1] + half + step / 2, ys[0] + half - step / 2))
        ax[1].set_xlim(0, img.shape[1]); ax[1].set_ylim(img.shape[0], 0); ax[1].set_aspect("equal")
        ax[1].set_title(f"Similarity map (−D3), window centers, {bh_} px windows"); plt.colorbar(im, ax=ax[1], fraction=0.035)
        plt.tight_layout(); plt.savefig(f"{od}/{fname}", dpi=110); plt.close()
        return dets
    heat(photo, PHOTO_GT, "photo_heatmap.png", "Top detection (HSV, D3) in green; ground truth in magenta")

    # ---- C. sign sheet: recognition (rank all 18 signs against the STOP template)
    stop_gt = SHEET_SIGNS["STOP"]
    log(f"\nC) SIGN SHEET {sheet_path}: per-sign distance to the STOP template (box = sign, template resized to the box)")
    R["sheet_rank"] = {}
    for sp in ["rgb", "hsv"]:
        for grid in [1, 3, 5]:
            for m in ["d1", "d3"]:
                ds = {}
                for name, box in SHEET_SIGNS.items():
                    f, t, n = _sign_features(sheet, box, tmpl, sp, grid, False, tmask)
                    ds[name] = float(METRICS[m](f, t) / n)
                order = sorted(ds, key=ds.get)
                key = f"{sp}-g{grid}-{m}"
                R["sheet_rank"][key] = ds
                gap = ds[order[1]] / ds[order[0]] if order[0] == "STOP" else ds["STOP"] / ds[order[0]]
                log(f"   {key:10} STOP rank {order.index('STOP') + 1:2d}  top-4: " +
                    ", ".join(f"{o} {ds[o]:.3f}" for o in order[:4]) + f"   ratio 2nd/1st {gap:.2f}")

    # ---- D. sign sheet: sliding-window detection
    log("\nD) SIGN SHEET sliding-window detection, no mask (best window IoU with STOP; hits above threshold)")
    R["sheet_det"] = {}
    for sp in ["gray", "rgb", "hsv", "lab"]:
        cells = []
        for m in ["d1", "d3"]:
            r = _eval(sheet, tmpl, stop_gt, space=sp, metric=m)
            hit_names = []
            for _, bb in r["dets"]:
                hit_names.append(max(SHEET_SIGNS, key=lambda k: iou(SHEET_SIGNS[k], bb)) if max(iou(v, bb) for v in SHEET_SIGNS.values()) > 0 else "background")
            R["sheet_det"][f"{sp}-{m}"] = dict(iou=r["iou"], box=r["box"], d=r["d"], passed=r["passed"], hits=hit_names[:5],
                                               hit_d=[float(dd) for dd, _ in r["dets"][:5]])
            cells.append(f"{m}: IoU {r['iou']:.2f}{'*' if r['passed'] else ' '} hits {hit_names[:4]}")
        log(f"   {sp:5} " + " | ".join(cells))
    dets = heat(sheet, stop_gt, "sheet_heatmap.png", "Top-5 detections (HSV, D3); STOP ground truth in magenta", n_show=5)

    # per-sign bar chart (HSV, D3, grid 1 vs grid 3)
    names = list(SHEET_SIGNS)
    g1, g3 = R["sheet_rank"]["hsv-g1-d3"], R["sheet_rank"]["hsv-g3-d3"]
    order = sorted(names, key=g3.get)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    xx = np.arange(len(order))
    ax.bar(xx - 0.2, [g1[n] for n in order], 0.4, color="#bfbfbf", label="color only (1 × 1 cell)")
    ax.bar(xx + 0.2, [g3[n] for n in order], 0.4, color=["#c0392b" if n == "STOP" else "#2c3e50" for n in order], label="3 × 3 grid")
    ax.set_xticks(xx, order, rotation=55, ha="right", fontsize=9); ax.set_ylabel("D3 to STOP template (lower = closer)")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False)
    plt.tight_layout(); plt.savefig(f"{od}/sheet_ranking.png", dpi=130); plt.close()

    # ---- E. photo: slide-6 features per region, local maps
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    regions = {"whole photo": (0, 0, gray.shape[1], gray.shape[0]), "stop sign": PHOTO_GT, "road": (150, 320, 66, 66),
               "grass": (400, 225, 66, 60), "sky": (650, 10, 66, 40), "house siding": (330, 160, 66, 50), "red-leaf tree": (300, 30, 66, 66)}
    log("\nE) PHOTO histogram features per region (gray levels 0-255)")
    R["regions"] = {}
    for name, (rx, ry, rw, rh) in regions.items():
        hh = np.bincount(gray[ry:ry + rh, rx:rx + rw].ravel(), minlength=256)
        st = hist_stats(hh); p, r_ = hh / hh.sum(), np.arange(256)
        R["regions"][name] = dict(mean=st["mean"], var=st["variance"], contrast=st["contrast"], u3=st["u3"], entropy=st["entropy"],
                                  dark=hist_area(p, r_, 0, 63, hh.sum()), n=int(hh.sum()))
        log(f"   {name:14} mean {st['mean']:6.1f} var {st['variance']:8.1f} contrast {st['contrast']:5.1f} u3 {st['u3']:10.0f} entropy {st['entropy']:.2f} area[0,63] {R['regions'][name]['dark']:.0f}/{hh.sum()}")
    m_map, v_map = local_mean_var(gray, 9)
    ih = IntegralHistogram(gray[..., None], [0], [32], [256])
    H, W = gray.shape; k = 9
    yy, xx2 = np.mgrid[0:H, 0:W]
    y0, y1 = np.clip(yy - k // 2, 0, H), np.clip(yy + k // 2 + 1, 0, H)
    x0, x1 = np.clip(xx2 - k // 2, 0, W), np.clip(xx2 + k // 2 + 1, 0, W)
    hloc = ih.ii[y1, x1] - ih.ii[y0, x1] - ih.ii[y1, x0] + ih.ii[y0, x0]
    ent = hist_entropy(hloc / hloc.sum(-1, keepdims=True))
    fig, ax = plt.subplots(1, 4, figsize=(18, 2.9))
    for a, img, t, cm in zip(ax, [gray, m_map, np.sqrt(v_map), ent],
                             ["Grayscale f(x,y)", "Local mean $m_{S_{xy}}$ (9×9)", "Local std $\\sigma_{S_{xy}}$ (contrast)", "Local entropy (32 bins, bits)"],
                             ["gray", "gray", "magma", "viridis"]):
        im = a.imshow(img, cmap=cm); a.set_title(t); a.axis("off"); plt.colorbar(im, ax=a, fraction=0.025)
    plt.tight_layout(); plt.savefig(f"{od}/photo_local_stats.png", dpi=110); plt.close()

    # ---- F. photo: do the slide-6 stats help detection? (no mask)
    log("\nF) PHOTO detection with stats appended (no mask): IoU of best window")
    R["stats"] = {}
    for sp in spaces:
        for m in metrics:
            row = []
            for wgt in [0, 0.5, 1, 2]:
                dets_, maps_, _ = detect(photo, tmpl, space=sp, metric=m, stats_w=wgt)
                row.append(iou(_best_window(maps_)[1], PHOTO_GT))
            R["stats"][f"{sp}-{m}"] = row
            log(f"   {sp:5} {m}  " + "  ".join(f"{v:.2f}" for v in row))

    # ---- G. color space on the sign sheet: STOP color model, backprojection, AUC
    sx, sy, sw, sh = stop_gt
    sheet_hsv = cv2.cvtColor(sheet, cv2.COLOR_BGR2HSV)
    redm = ((sheet_hsv[..., 0] < 10) | (sheet_hsv[..., 0] > 170)) & (sheet_hsv[..., 1] > 100)
    stop_red = np.zeros(redm.shape, bool); stop_red[sy:sy + sh, sx:sx + sw] = redm[sy:sy + sh, sx:sx + sw]
    away = np.ones_like(stop_red); away[sy:sy + sh, sx:sx + sw] = False
    other_red = away & redm
    cal = np.zeros_like(stop_red); cal[sy + 30:sy + 70, sx + 100:sx + 200] = True     # red band above "STOP"
    log("\nG) SIGN SHEET color-only (backprojection) AUC: STOP red pixels vs. everything else")
    R["sheet_auc"] = {}
    hdr = f"   {'':15}" + "".join(f"{k:>14}" for k in CS_LIGHTS)
    log(hdr)
    for spn in CS_SPACES:
        hs = _cs_model([sheet], spn, [cal])
        vals = []
        for L in CS_LIGHTS.values():
            bp = _cs_backproject(_relight(sheet, *L), hs, spn)
            vals.append(_auc(bp[stop_red], bp[away]))
        R["sheet_auc"][spn] = vals
        log(f"   {spn:15}" + "".join(f"{v:14.3f}" for v in vals))
    other_red = np.zeros_like(stop_red)                                  # red pixels of the other red signs
    for n in ["YIELD", "DO NOT ENTER", "INTERSTATE 95", "NO U-TURN"]:
        bx, by, bw, bh = SHEET_SIGNS[n]; other_red[by:by + bh, bx:bx + bw] |= redm[by:by + bh, bx:bx + bw]
    other_red &= ~stop_red
    R["sheet_red_vs_red"] = {}
    for spn in CS_SPACES:
        bp = _cs_backproject(sheet, _cs_model([sheet], spn, [cal]), spn)
        R["sheet_red_vs_red"][spn] = _auc(bp[stop_red], bp[other_red])
    log("   STOP red vs. red of YIELD / DO NOT ENTER / INTERSTATE / NO U-TURN (normal light): " +
        ", ".join(f"{k} {v:.3f}" for k, v in R["sheet_red_vs_red"].items()))

    rows = [("original light", CS_LIGHTS["normal"]), ("dimmed ×0.45", CS_LIGHTS["dim x0.45"])]
    fig, ax = plt.subplots(2, len(CS_SPACES) + 1, figsize=(3 * (len(CS_SPACES) + 1), 6.2))
    for r_, (lab, L) in enumerate(rows):
        lit = _relight(sheet, *L); disp = cv2.cvtColor(lit, cv2.COLOR_BGR2RGB).copy()
        if r_ == 0:
            cv2.rectangle(disp, (sx + 100, sy + 30), (sx + 200, sy + 70), (0, 200, 255), 4)
        ax[r_, 0].imshow(disp); ax[r_, 0].set_title(lab, fontsize=11); ax[r_, 0].axis("off")
        for c_, spn in enumerate(CS_SPACES):
            hs = _cs_model([sheet], spn, [cal]); bp = _cs_backproject(lit, hs, spn)
            ax[r_, c_ + 1].imshow(bp / max(bp.max(), 1e-12), cmap="inferno", vmin=0, vmax=1)
            ax[r_, c_ + 1].set_title(f"{spn}  AUC = {R['sheet_auc'][spn][list(CS_LIGHTS.values()).index(L)]:.2f}", fontsize=11); ax[r_, c_ + 1].axis("off")
    plt.tight_layout(); plt.savefig(f"{od}/sheet_backproj.png", dpi=110); plt.close()

    # scatter: skin vs photo stop-sign red vs backgrounds
    rng = np.random.default_rng(3)
    mst = [_hex_bgr(c) for c in MST_HEX]
    bg = [_cs_patch(_hex_bgr(c), L, rng) for c in CS_BG.values() for L in CS_LIGHTS.values()]
    pm = cv2.cvtColor(photo, cv2.COLOR_BGR2HSV)
    pred = ((pm[..., 0] < 10) | (pm[..., 0] > 170)) & (pm[..., 1] > 120) & (pm[..., 2] > 80)
    psign = np.zeros(pred.shape, bool); psign[y:y + h, x:x + w] = pred[y:y + h, x:x + w]
    sub = lambda img, k=60: img.reshape(-1, 3)[rng.choice(img.size // 3, min(k, img.size // 3), replace=False)].reshape(-1, 1, 3)
    skin_pts = np.concatenate([sub(_cs_patch(c, L, rng)) for c in mst for L in CS_LIGHTS.values()])
    bg_pts = np.concatenate([sub(b) for b in bg])
    sl_pts = np.concatenate([sub(_cs_patch(_hex_bgr(v), L, rng)) for v in CS_SKINLIKE.values() for L in CS_LIGHTS.values()])
    sign_pts = np.concatenate([sub(_relight(photo, *L)[psign].reshape(-1, 1, 3), 150) for L in CS_LIGHTS.values()])

    _cs_scatter([(bg_pts, "#9aa5b1", "other background"), (sl_pts, "#8d6e63", "wood / beige / cardboard"),
                 (sign_pts, "#d62728", "stop-sign red (photo)"), (skin_pts, "#f2a65a", "skin, all tones x lights")],
                "Where stop-sign red and skin fall in each color space", f"{od}/colorspace_scatter_photo.png", rng)

    json.dump(R, open(f"{od}/realtest.json", "w"), indent=1, default=float)
    open(f"{od}/realtest_results.txt", "w").write("\n".join(lines) + "\n")


# -------------------- `detect` command

def run_detect(a):

    scene = cv2.imread(a.scene)
    tmpl = cv2.imread(a.template, cv2.IMREAD_UNCHANGED)
    tmask = None
    if tmpl.ndim == 3 and tmpl.shape[2] == 4:            # alpha channel -> template mask
        tmask, tmpl = tmpl[..., 3] > 0, tmpl[..., :3]
    dets, _, thr = detect(scene, tmpl, a.space, a.metric, a.grid, stride=a.stride,
                          template_mask=tmask,
                          thresh=a.thresh, blur=a.blur, ksize=a.ksize, use_v=a.use_v,
                          top_k=a.top_k, stats_w=a.stats_w)
    print(f"threshold={thr:.4f}")
    for i, (d, b) in enumerate(dets):
        print(f"#{i+1}  dist={d:.4f}  box(x,y,w,h)={b}")
    cv2.imwrite(a.out, draw(scene, dets))


# -------------------- Figure: red-sign histograms vs the template

def fig_sign_hists(sheet, tmpl, path, names=("STOP", "DO NOT ENTER", "YIELD")):
    """Whole-sign HSV histograms of the red signs vs the template; D3 with 1 cell and with the 3 x 3 grid."""
    res = {}
    hs, labs, cols = [], [], ["#7f7f7f", "#c0392b", "#8e44ad", "#d68910"]
    for n in names:
        b = SHEET_SIGNS[n]; t = cv2.resize(tmpl, (b[2], b[3]), interpolation=cv2.INTER_AREA)
        h1, r1 = region_hists(sheet, b, "hsv"), region_hists(t, None, "hsv")
        h3, r3 = region_hists(sheet, b, "hsv", grid=3), region_hists(t, None, "hsv", grid=3)
        res[n] = (hist_dist(h1, r1), hist_dist(h3, r3))
        if not hs:
            hs.append(r1); labs.append("Template")
        hs.append(h1); labs.append(f"{n}: D3 {res[n][0]:.3f} (1 cell), {res[n][1]:.3f} (3×3)")
    fig, ax = plt.subplots(figsize=(10, 2.8))
    _bars(ax, hs, labs, cols, HSV_GROUPS)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.36, 1.0), frameon=False); ax.set_ylabel("share of pixels", fontsize=8)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    return res


# ==============================================================================
# APPLICATION: HAND DETECTION
# Skin tones (Monk Skin Tone scale, a real portrait) and Test 3 (three ASL frames).
# Skin histogram -> backprojection -> blur + Otsu threshold; channels compared.
# ==============================================================================


# -------------------- Skin vs stop-sign red (`colorspace` command)

# Real-skin photo: Adobe Stock portrait (540 x 360), hand-drawn ground truth
SKIN_PHOTO = "skin_photo.jpg"
SKIN_PHOTO_GT = {"face_ellipse": ((306, 62), (40, 44)),            # bald head + face, above the beard
                 "hand": [(237, 262), (255, 257), (262, 272), (258, 290), (240, 292), (235, 276)],
                 "head_ellipse": ((306, 85), (62, 78)),             # head, beard, ears, neck (excluded from "rest")
                 "cheek": (321, 78, 15, 15)}                       # x, y, w, h calibration patch


# Monk Skin Tone scale MST-1..10 (hex RGB), light -> dark
MST_HEX = ["f6ede4", "f3e7db", "f7ead0", "eadaba", "d7bd96", "a07e56", "825c43", "604134", "3a312a", "292420"]


CS_BG = {"red object": "c02020", "green plant": "3c7a32", "blue jeans": "34507a", "white paper": "f0f0f0",
         "gray floor": "707070", "dark shirt": "202020", "orange": "e07020"}


CS_SKINLIKE = {"wood table": "8b5a2b", "beige wall": "d8c8b0", "cardboard": "b08850"}


def _cs_patch(bgr, light, rng, n=48, tex=0.06, noise=6):
    """Flat color patch with texture, lighting and sensor noise."""
    gain, tint = light
    img = bgr * (1 + rng.normal(0, tex, (n, n, 1))) * gain * np.array(tint) + rng.normal(0, noise, (n, n, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def _cs_scatter(groups, title, fname, rng):
    """groups: [(Nx1x3 BGR pixels, color, label)]. One panel per color space; Gray as a strip plot."""
    labels = {"RGB": ("R", "G"), "HSV (H,S)": ("Hue (0-180)", "Saturation"), "Lab (a*,b*)": ("a* (+128)", "b* (+128)"), "YCrCb (Cr,Cb)": ("Cr", "Cb")}
    fig, ax = plt.subplots(1, len(CS_SPACES), figsize=(5.2 * len(CS_SPACES), 5.2))
    for a_, spn in zip(ax, CS_SPACES):
        conv, chs, _, _ = CS_SPACES[spn]
        for z, (P, c, lab) in enumerate(groups):
            x = P if conv is None else cv2.cvtColor(P, conv)
            if spn == "Gray":
                u = x.reshape(-1).astype(float); v = z + rng.uniform(-0.3, 0.3, len(u))
            else:
                x = x.reshape(-1, 3).astype(float)
                if conv is None:
                    x = x[:, ::-1]; u, v = x[:, 0], x[:, 1]
                else:
                    u, v = x[:, chs[0]], x[:, chs[1]]
            a_.scatter(u, v, s=5, c=c, alpha=0.45, label=lab, zorder=z + 1, edgecolors="none")
        a_.set_title(spn)
        if spn == "Gray":
            a_.set_xlabel("Gray level"); a_.set_yticks(range(len(groups)), [g[2] for g in groups], fontsize=8)
        else:
            a_.set_xlabel(labels[spn][0]); a_.set_ylabel(labels[spn][1])
    rgb_i = list(CS_SPACES).index("RGB")
    ax[rgb_i].plot([0, 255], [0, 255], "k:", lw=0.8); ax[rgb_i].legend(loc="upper left", markerscale=3, fontsize=8)
    plt.suptitle(title); plt.tight_layout(); plt.savefig(fname, dpi=100); plt.close()


def run_colorspace():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(3)
    lines = []
    log = lambda t="": (print(t), lines.append(t))
    names = list(CS_SPACES)
    hdr = lambda: log(f"   {'':15}" + "".join(f"{k:>14}" for k in CS_LIGHTS))

    # ---- 1. stop sign (synthetic scene): calibrate on a red patch of the sign
    scene, gt = scene_img()
    tmpl, tmask = stop_sign(120)
    x, y, w, h = gt
    t_small = cv2.resize(tmpl, (w, h), interpolation=cv2.INTER_NEAREST)
    m_small = cv2.resize(tmask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    red = np.zeros(scene.shape[:2], bool)
    red[y:y + h, x:x + w] = m_small & (t_small[..., 2].astype(int) > t_small[..., 1].astype(int) + 80)
    away = np.ones_like(red); away[y - 5:y + h + 5, x - 5:x + w + 5] = False
    cal = np.zeros_like(red); cal[y + 8:y + 20, x + 15:x + 43] = True
    sc_hsv = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
    nonred = away & ~(((sc_hsv[..., 0] < 10) | (sc_hsv[..., 0] > 170)) & (sc_hsv[..., 1] > 100))
    log("1) STOP SIGN - pixel AUC, sign red vs. rest of scene (calibrated on a patch of the sign)")
    log("   a) vs. ALL background (includes red roof / mailbox / car = same red)")
    hdr()
    stop_auc = {}
    for s in names:
        hs = _cs_model([scene], s, [cal])
        vals = []
        for L in CS_LIGHTS.values():
            bp = _cs_backproject(_relight(scene, *L), hs, s)
            vals.append(_auc(bp[red], bp[away]))
        stop_auc[s] = vals
        log(f"   {s:15}" + "".join(f"{v:14.3f}" for v in vals))
    log("   b) vs. NON-RED background only")
    hdr()
    for s in names:
        hs = _cs_model([scene], s, [cal])
        vals = [_auc(*(lambda bp: (bp[red], bp[nonred]))(_cs_backproject(_relight(scene, *L), hs, s)))
                for L in CS_LIGHTS.values()]
        log(f"   {s:15}" + "".join(f"{v:14.3f}" for v in vals))

    # ---- 2. skin, synthetic: 10 Monk tones x 5 lights
    mst = [_hex_bgr(c) for c in MST_HEX]
    bg = [_cs_patch(_hex_bgr(c), L, rng) for c in CS_BG.values() for L in CS_LIGHTS.values()]
    log("\n2) SKIN (synthetic, Monk Skin Tone 1-10) - pixel AUC skin vs. non-skin background")
    log("   A) generic skin model (trained on all tones x lights, tested on fresh patches)")
    log(f"   {'':15}{'AUC':>8}   skin-like materials scored above the 95th pct of background")
    for s in names:
        hs = _cs_model([_cs_patch(c, L, rng) for c in mst for L in CS_LIGHTS.values()], s)
        neg = np.concatenate([_cs_backproject(b, hs, s).ravel() for b in bg])
        pos = np.concatenate([_cs_backproject(_cs_patch(c, L, rng), hs, s).ravel()
                              for c in mst for L in CS_LIGHTS.values()])
        t = np.quantile(neg, 0.95)
        sk = "  ".join(f"{k}={np.mean([(_cs_backproject(_cs_patch(_hex_bgr(v), L, rng), hs, s) >= t).mean() for L in CS_LIGHTS.values()]):.2f}"
                       for k, v in CS_SKINLIKE.items())
        log(f"   {s:15}{_auc(pos, neg):8.3f}   {sk}")
    log("   B) one-person calibration (each tone trained under normal light, tested under each light; mean of 10 tones)")
    hdr()
    for s in names:
        res = {k: [] for k in CS_LIGHTS}
        for c in mst:
            hs = _cs_model([_cs_patch(c, CS_LIGHTS["normal"], rng)], s)
            neg = np.concatenate([_cs_backproject(b, hs, s).ravel() for b in bg])
            for k, L in CS_LIGHTS.items():
                res[k].append(_auc(_cs_backproject(_cs_patch(c, L, rng), hs, s).ravel(), neg))
        log(f"   {s:15}" + "".join(f"{np.mean(v):14.3f}" for v in res.values()))

    # ---- 3. skin, real photo (Adobe Stock portrait, skin_photo.jpg): calibrate on a cheek patch
    photo = cv2.imread(SKIN_PHOTO)
    if photo is None:
        log(f"\n3) real photo skipped: could not read {SKIN_PHOTO}")
    face_auc = {}
    if photo is not None:
        g = SKIN_PHOTO_GT
        skin = np.zeros(photo.shape[:2], np.uint8)
        cv2.ellipse(skin, *g["face_ellipse"], 0, 0, 360, 1, -1)
        cv2.fillPoly(skin, [np.array(g["hand"])], 1)
        band = cv2.dilate(skin, np.ones((15, 15), np.uint8)) - cv2.erode(skin, np.ones((15, 15), np.uint8))
        head = np.zeros_like(skin); cv2.ellipse(head, *g["head_ellipse"], 0, 0, 360, 1, -1)   # beard, ears, neck: ignored
        face, rest = (skin & ~band).astype(bool), ~(skin | band | head).astype(bool)
        x, y, w, h = g["cheek"]
        cheek = np.zeros(photo.shape[:2], bool); cheek[y:y + h, x:x + w] = True
        log("\n3) SKIN (real photo, Adobe Stock portrait) - pixel AUC skin (face, head, hand) vs. rest (denim, beard, street, buildings)")
        log(f"   calibrated on a {w}x{h} cheek patch under original lighting")
        hdr()
        for s in names:
            hs = _cs_model([photo], s, [cheek])
            vals = []
            for L in CS_LIGHTS.values():
                bp = _cs_backproject(_relight(photo, *L), hs, s)
                vals.append(_auc(bp[face], bp[rest]))
            face_auc[s] = vals
            log(f"   {s:15}" + "".join(f"{v:14.3f}" for v in vals))
    log("\n   AUC: 1.0 = perfect separation, 0.5 = chance.")
    open(f"{OUT}/colorspace_results.txt", "w").write("\n".join(lines) + "\n")

    # ---- figure 1: where the colors live in each space
    sub = lambda img, k=60: img.reshape(-1, 3)[rng.choice(img.size // 3, k, replace=False)].reshape(-1, 1, 3)
    skin_pts = np.concatenate([sub(_cs_patch(c, L, rng)) for c in mst for L in CS_LIGHTS.values()])
    bg_pts = np.concatenate([sub(b) for b in bg])
    sl_pts = np.concatenate([sub(_cs_patch(_hex_bgr(v), L, rng)) for v in CS_SKINLIKE.values() for L in CS_LIGHTS.values()])
    sign_pts = np.concatenate([sub(_relight(scene, *L)[red].reshape(-1, 1, 3), 150) for L in CS_LIGHTS.values()])
    _cs_scatter([(bg_pts, "#9aa5b1", "other background"), (sl_pts, "#8d6e63", "wood / beige / cardboard"),
                 (sign_pts, "#d62728", "stop-sign red"), (skin_pts, "#f2a65a", "skin, all tones x lights")],
                "Where the colors fall in each color space", f"{OUT}/colorspace_scatter.png", rng)

    # ---- figure 2: skin backprojection on the real photo, original and dimmed
    if photo is not None:
        rows = [("original light", CS_LIGHTS["normal"]), ("dimmed ×0.45", CS_LIGHTS["dim x0.45"])]
        fig, ax = plt.subplots(2, len(names) + 1, figsize=(3 * (len(names) + 1), 2 * 3 * photo.shape[0] / photo.shape[1] + 0.9))
        for r_, (lab, L) in enumerate(rows):
            lit = _relight(photo, *L); disp = cv2.cvtColor(lit, cv2.COLOR_BGR2RGB).copy()
            if r_ == 0:
                ys, xs = np.where(cheek); cv2.rectangle(disp, (xs.min(), ys.min()), (xs.max(), ys.max()), (0, 255, 255), 2)
            ax[r_, 0].imshow(disp); ax[r_, 0].set_title(lab, fontsize=11); ax[r_, 0].axis("off")
            for c_, s in enumerate(names):
                bp = _cs_backproject(lit, _cs_model([photo], s, [cheek]), s)
                ax[r_, c_ + 1].imshow(bp / max(bp.max(), 1e-12), cmap="inferno", vmin=0, vmax=1)
                ax[r_, c_ + 1].set_title(f"{s}  AUC = {face_auc[s][list(CS_LIGHTS.values()).index(L)]:.2f}", fontsize=11); ax[r_, c_ + 1].axis("off")
        plt.tight_layout(); plt.savefig(f"{OUT}/colorspace_backproj_photo.png", dpi=110); plt.close()


# -------------------- Test 3: hands in ASL frames (`handtest` command)

# Hand-drawn ground truth: skin polygons (hands, arms, neck), face, hand box, ignore regions.
HAND_GT = {
    "f1": {"source": "How2Sign", "hand_box": (270, 512, 165, 160), "face_ellipse": ((370, 250), (50, 70)),
           "skin": [[(335, 300), (410, 300), (418, 362), (330, 362)],
                    [(135, 525), (200, 515), (275, 535), (290, 605), (200, 610), (135, 602)],
                    [(275, 535), (380, 515), (430, 555), (425, 640), (370, 668), (300, 650), (280, 600)],
                    [(420, 560), (470, 548), (538, 532), (545, 600), (480, 612), (425, 615)]], "ignore": []},
    "f2": {"source": "How2Sign", "hand_box": (255, 303, 82, 110), "face_ellipse": ((360, 252), (50, 72)),
           "skin": [[(325, 300), (410, 300), (440, 330), (400, 390), (370, 420), (335, 390), (320, 330)],
                    [(258, 330), (290, 305), (335, 310), (335, 360), (320, 395), (290, 410), (260, 400), (255, 360)],
                    [(195, 470), (240, 400), (275, 405), (260, 440), (235, 500), (195, 500)],
                    [(385, 690), (440, 675), (480, 680), (500, 700), (480, 720), (400, 720)],
                    [(495, 620), (540, 610), (560, 650), (545, 720), (500, 720), (500, 680)]], "ignore": []},
    "f3": {"source": "YouTube ASL video", "hand_box": (130, 285, 310, 405),
           "face_poly": [(240, 0), (530, 0), (525, 90), (490, 150), (440, 200), (380, 215), (320, 200), (275, 160), (250, 100)],
           "skin": [[(285, 190), (470, 190), (495, 235), (400, 260), (290, 240)],
                    [(250, 330), (300, 285), (380, 288), (440, 320), (430, 345), (380, 360), (385, 410), (350, 420), (290, 410), (250, 380)],
                    [(140, 470), (200, 465), (250, 445), (300, 470), (330, 520), (370, 570), (420, 620), (420, 690), (330, 690), (250, 640), (170, 600), (130, 540)]],
           "ignore": [[(60, 380), (250, 370), (260, 440), (150, 480), (60, 470)]]},
}


def _hand_masks(g, shape, band=17):
    """skin (incl. face), face, ignore (edge band of +-8 px around every outline + ignore polygons)."""
    skin = np.zeros(shape[:2], np.uint8); face = np.zeros_like(skin)
    if g.get("face_ellipse"):
        c, a = g["face_ellipse"]; cv2.ellipse(face, c, a, 0, 0, 360, 1, -1)
    if g.get("face_poly"):
        cv2.fillPoly(face, [np.array(g["face_poly"])], 1)
    for p in g["skin"]:
        cv2.fillPoly(skin, [np.array(p)], 1)
    skin |= face
    ign = np.zeros_like(skin)
    for p in g["ignore"]:
        cv2.fillPoly(ign, [np.array(p)], 1)
    k = np.ones((band, band), np.uint8)
    edge = cv2.dilate(skin, k) - cv2.erode(skin, k)
    return skin.astype(bool), face.astype(bool), (edge | ign).astype(bool)


def _skin_model(imgs, masks, space, smooth=1.0):
    """Skin color histogram from calibration pixels, Gaussian-smoothed over bins."""
    from scipy.ndimage import gaussian_filter
    h = _cs_model(imgs, space, masks)
    return gaussian_filter(h, smooth) if smooth else h


def _skin_mask(bp):
    """Skin mask from a backprojection: Gaussian blur (step 1B) + Otsu threshold (histogram-based)."""
    bp8 = cv2.GaussianBlur((255 * bp / max(bp.max(), 1e-12)).astype(np.uint8), (7, 7), 0)
    _, m = cv2.threshold(bp8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m > 0


def _inner(box, shape):
    x, y, w, h = box
    m = np.zeros(shape[:2], bool); m[y + h // 5:y + 4 * h // 5, x + w // 5:x + 4 * w // 5] = True
    return m


def run_handtest(paths):
    import json
    od = os.path.join(OUT, "hands"); os.makedirs(od, exist_ok=True)
    keys = list(HAND_GT)
    F = {k: cv2.imread(p) for k, p in zip(keys, paths)}
    if any(v is None for v in F.values()):
        raise SystemExit(f"could not read one of {paths}")
    M = {k: _hand_masks(HAND_GT[k], F[k].shape) for k in keys}
    names = list(CS_SPACES)
    R, lines = {}, []
    log = lambda t="": (print(t), lines.append(t))

    # A. color only: skin model from each frame's annotated face, applied to every frame
    ann_face = {}
    for k in keys:
        fm = cv2.erode(M[k][1].astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)
        fm[:30] = False
        ann_face[k] = fm
    log("\nA) pixel AUC, hands/arms/neck vs non-skin (face excluded). Model: smoothed histogram of one frame's face.")
    log(f"   {'space':14} {'model':6} " + "  ".join(f"{k:>6}" for k in keys) + "   same-person  cross-person")
    R["auc"] = {}
    for s in names:
        same, cross = [], []
        for src in keys:
            h = _skin_model([F[src]], [ann_face[src]], s)
            row = []
            for k in keys:
                skin, face, ign = M[k]
                bp = _cs_backproject(F[k], h, s)
                a = _auc(bp[skin & ~face & ~ign], bp[~skin & ~ign])
                row.append(a); (same if k == src else cross).append(a)
                R["auc"][f"{s}|{src}|{k}"] = a
            log(f"   {s:14} {src:6} " + "  ".join(f"{v:6.3f}" for v in row))
        R["auc"][f"{s}|same"] = float(np.mean(same)); R["auc"][f"{s}|cross"] = float(np.mean(cross))
        log(f"   {s:14} {'mean':6} {'':22}   {np.mean(same):11.3f}  {np.mean(cross):12.3f}")

    # what lights up in f3 with its own face model
    log("\n   f3, own-face model: mean response (fraction of max) on hands vs. distractors")
    R["f3_regions"] = {}
    for s in names:
        h = _skin_model([F["f3"]], [ann_face["f3"]], s); bp = _cs_backproject(F["f3"], h, s); mx = bp.max()
        skin, face, ign = M["f3"]
        vals = {"hands/arms": bp[skin & ~face & ~ign].mean() / mx}
        for nm_, (x, y, w, hh) in {"beige wall": (0, 0, 200, 250), "maroon shirt": (500, 300, 200, 150), "yellow caption": (0, 660, 720, 60)}.items():
            vals[nm_] = bp[y:y + hh, x:x + w].mean() / mx
        R["f3_regions"][s] = vals
        log(f"   {s:14} " + "  ".join(f"{a} {v:.2f}" for a, v in vals.items()))

    # B. skin masks: own-face model -> backprojection -> Gaussian blur -> Otsu (no other post-processing)
    log("\nB) skin masks by Otsu thresholding of the backprojection (model from the frame's own face)")
    log("   mask IoU = predicted vs annotated hands/arms/neck (face and edge band excluded)")
    R["masks"] = {}
    pipe = {}
    for s in names:
        row = []
        for k in keys:
            h = _skin_model([F[k]], [ann_face[k]], s)
            bp = _cs_backproject(F[k], h, s)
            pred = _skin_mask(bp)
            skin, face, ign = M[k]; gt = skin & ~face; valid = ~ign & ~face
            seg = (pred & gt & valid).sum() / max(((pred | gt) & valid).sum(), 1)
            pipe[(s, k)] = pred
            R["masks"][f"{s}|{k}"] = float(seg)
            row.append(f"{k}: {seg:.2f}")
        R["masks"][f"{s}|mean"] = float(np.mean([R["masks"][f"{s}|{k}"] for k in keys]))
        log(f"   {s:14} " + " | ".join(row) + f" | mean {R['masks'][f'{s}|mean']:.2f}")

    # C. histogram detector with a hand template (frame 2's raised hand) -> other frames
    log("\nC) Part-1 detector with a hand template (f2 raised hand box), top-1 IoU with each hand box")
    x, y, w, h = HAND_GT["f2"]["hand_box"]
    tmpl = F["f2"][y:y + h, x:x + w].copy()
    R["template"] = {}
    for sp in ["gray", "rgb", "hsv", "lab", "ycrcb"]:
        row = []
        for k in keys:
            dets, maps, thr = detect(F[k], tmpl, space=sp, metric="d3", top_k=5)
            d, b = _best_window(maps)
            hit = iou(b, HAND_GT[k]["hand_box"])
            onface = M[k][1][b[1] + b[3] // 2, b[0] + b[2] // 2]
            R["template"][f"{sp}|{k}"] = dict(iou=hit, box=b, on_face=bool(onface))
            row.append(f"{k}: IoU {hit:.2f}{' (face)' if onface else ''}")
        log(f"   {sp:6} " + " | ".join(row))


    # D. single channels vs. channel pairs (model from each frame's annotated face)
    def chan_imgs(img):
        b, g, r = cv2.split(img); h_, s_, v_ = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        l_, a_, b_ = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
        y_, cr, cb = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb))      # Y = OpenCV grayscale
        return {"R": (r, 256), "G": (g, 256), "B": (b, 256), "H": (h_, 180), "S": (s_, 256), "V": (v_, 256),
                "L": (l_, 256), "a": (a_, 256), "b": (b_, 256), "Y": (y_, 256), "Cr": (cr, 256), "Cb": (cb, 256)}
    SETS = {c: [c] for c in ["R", "G", "B", "H", "S", "V", "L", "a", "b", "Y", "Cr", "Cb"]}
    SETS.update({"H+S": ["H", "S"], "a+b": ["a", "b"], "Cr+Cb": ["Cr", "Cb"], "R+G+B": ["R", "G", "B"]})
    CI = {k: chan_imgs(F[k]) for k in keys}

    def cidx(k, chs, nb):
        return tuple(np.minimum(CI[k][c][0].astype(int) * nb // CI[k][c][1], nb - 1) for c in chs)

    def cmodel(k, mask, chs):
        from scipy.ndimage import gaussian_filter
        nb = 32 if len(chs) < 3 else 16
        h = np.zeros([nb] * len(chs)); np.add.at(h, tuple(a[mask] for a in cidx(k, chs, nb)), 1)
        return gaussian_filter(h / h.sum(), 1), nb

    log("\nD) single channels vs. combinations: pixel AUC (model from annotated face), mask IoU (own face model)")
    log(f"   {'chan':6} " + " ".join(f"{k:>5}" for k in keys) + "   same   cross | mask IoU " + " ".join(f"{k:>5}" for k in keys))
    R["channels"] = {}
    chan_masks = {}
    for nm_, chs in SETS.items():
        same, cross, segs = [], [], []
        for src in keys:
            h, nb = cmodel(src, ann_face[src], chs)
            for k in keys:
                skin, face, ign = M[k]
                bp = h[cidx(k, chs, nb)]
                a = _auc(bp[skin & ~face & ~ign], bp[~skin & ~ign]); (same if k == src else cross).append(a)
                if k == src:
                    pred = _skin_mask(bp)
                    gt = skin & ~face; valid = ~ign & ~face
                    segs.append((pred & gt & valid).sum() / max(((pred | gt) & valid).sum(), 1))
                    chan_masks[(nm_, k)] = pred
        R["channels"][nm_] = dict(same=same, same_mean=float(np.mean(same)), cross_mean=float(np.mean(cross)), seg=[float(v) for v in segs])
        log(f"   {nm_:6} " + " ".join(f"{v:5.2f}" for v in same) + f"  {np.mean(same):5.3f}  {np.mean(cross):5.3f} |          " +
            " ".join(f"{v:5.2f}" for v in segs))

    cols = ["R", "G", "B", "H", "S", "V", "L", "a", "b", "Y", "Cr", "Cb"]
    fig, ax = plt.subplots(3, len(cols) + 1, figsize=(26, 6.6))
    for r_, k in enumerate(keys):
        ax[r_, 0].imshow(cv2.cvtColor(F[k], cv2.COLOR_BGR2RGB)); ax[r_, 0].axis("off"); ax[r_, 0].set_title(k, fontsize=13)
        for c_, c in enumerate(cols):
            pred = chan_masks[(c, k)]
            vis = np.dstack([pred * 255] * 3).astype(np.uint8)
            vis[M[k][1]] = (vis[M[k][1]] * 0.3 + np.array([40, 90, 200]) * 0.7).astype(np.uint8)   # face shown in blue (removed)
            hx, hy, hw, hh = HAND_GT[k]["hand_box"]; cv2.rectangle(vis, (hx, hy), (hx + hw, hy + hh), (255, 215, 0), 6)
            ax[r_, c_ + 1].imshow(vis); ax[r_, c_ + 1].axis("off")
            ax[r_, c_ + 1].set_title(f"{c}  IoU {R['channels'][c]['seg'][r_]:.2f}", fontsize=13)
    plt.tight_layout(); plt.savefig(f"{od}/hands_channels.png", dpi=80); plt.close()

    # ---- figures
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))
    for a_, k in zip(ax, keys):
        skin, face, ign = M[k]
        ov = F[k].copy().astype(float)
        ov[skin & ~face & ~ign] = 0.45 * ov[skin & ~face & ~ign] + 0.55 * np.array([0, 140, 255])
        ov[face & ~ign] = 0.45 * ov[face & ~ign] + 0.55 * np.array([255, 160, 0])
        ov = ov.astype(np.uint8)
        hx, hy, hw, hh = HAND_GT[k]["hand_box"]; cv2.rectangle(ov, (hx, hy), (hx + hw, hy + hh), (0, 255, 255), 3)
        a_.imshow(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB)); a_.axis("off")
        a_.set_title(f"{k}: {HAND_GT[k]['source']}", fontsize=12)
    plt.tight_layout(); plt.savefig(f"{od}/hands_ground_truth.png", dpi=100); plt.close()

    fig, ax = plt.subplots(3, len(names) + 1, figsize=(3.75 * (len(names) + 1), 11.4))
    for r_, k in enumerate(keys):
        ax[r_, 0].imshow(cv2.cvtColor(F[k], cv2.COLOR_BGR2RGB)); ax[r_, 0].axis("off"); ax[r_, 0].set_title(k, fontsize=13)
        for c_, s in enumerate(names):
            pred = pipe[(s, k)]
            vis = np.dstack([pred * 255] * 3).astype(np.uint8)
            vis[M[k][1]] = (vis[M[k][1]] * 0.3 + np.array([40, 90, 200]) * 0.7).astype(np.uint8)   # face (not scored) in blue
            hx, hy, hw, hh = HAND_GT[k]["hand_box"]; cv2.rectangle(vis, (hx, hy), (hx + hw, hy + hh), (255, 215, 0), 4)
            ax[r_, c_ + 1].imshow(vis); ax[r_, c_ + 1].axis("off")
            ax[r_, c_ + 1].set_title(f"{s}: IoU {R['masks'][f'{s}|{k}']:.2f}", fontsize=13)
    plt.tight_layout(); plt.savefig(f"{od}/hands_masks.png", dpi=90); plt.close()

    fig, ax = plt.subplots(1, len(names) + 1, figsize=(4 * (len(names) + 1), 4.3))
    ax[0].imshow(cv2.cvtColor(F["f3"], cv2.COLOR_BGR2RGB)); ax[0].set_title("f3 (calibrated on his own face)", fontsize=12); ax[0].axis("off")
    for c_, s in enumerate(names):
        h = _skin_model([F["f3"]], [ann_face["f3"]], s); bp = _cs_backproject(F["f3"], h, s)
        ax[c_ + 1].imshow(bp / bp.max(), cmap="inferno", vmin=0, vmax=1); ax[c_ + 1].axis("off")
        ax[c_ + 1].set_title(f"{s}  AUC {R['auc'][f'{s}|f3|f3']:.2f}", fontsize=12)
    plt.tight_layout(); plt.savefig(f"{od}/hands_f3_backproj.png", dpi=100); plt.close()

    json.dump(R, open(f"{od}/handtest.json", "w"), indent=1, default=lambda o: o if not isinstance(o, np.generic) else o.item())
    open(f"{od}/handtest_results.txt", "w").write("\n".join(lines) + "\n")


# -------------------- Figures: skin vs background histograms, Otsu on the score histogram

def fig_skin_channel_hists(frame, gt, path):
    """1-D histograms of skin vs non-skin pixels (face excluded) in one ASL frame, per channel."""
    skin, face, ign = _hand_masks(gt, frame.shape)
    pos, neg = skin & ~face & ~ign, ~skin & ~ign
    conv = {"Gray": (cv2.COLOR_BGR2GRAY, None), "R (RGB)": (None, 2), "H (HSV)": (cv2.COLOR_BGR2HSV, 0),
            "Cr (YCrCb)": (cv2.COLOR_BGR2YCrCb, 1), "Cb (YCrCb)": (cv2.COLOR_BGR2YCrCb, 2)}
    fig, axes = plt.subplots(1, len(conv), figsize=(12, 2.3))
    res = {}
    for ax, (name, (c, k)) in zip(axes, conv.items()):
        x = frame if c is None else cv2.cvtColor(frame, c)
        v = x if k is None else x[..., k]
        top = 180 if name.startswith("H") else 256
        b = np.linspace(0, top, 33)
        hp, _ = np.histogram(v[pos], b); hn, _ = np.histogram(v[neg], b)
        hp, hn = hp / hp.sum(), hn / hn.sum()
        res[name] = float(np.minimum(hp, hn).sum())          # overlap: 0 = separate, 1 = identical
        ax.bar(b[:-1], hn, np.diff(b), align="edge", color="#7f8c8d", alpha=0.6, label="not skin", linewidth=0)
        ax.bar(b[:-1], hp, np.diff(b), align="edge", color="#e67e22", alpha=0.7, label="skin (hands, arms)", linewidth=0)
        ax.set_title(f"{name}: overlap {res[name]:.2f}", fontsize=9); ax.tick_params(labelsize=7); ax.set_yticks([])
    axes[0].legend(fontsize=7, frameon=False)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    return res


def fig_otsu_hists(frame, gt, path):
    """Histogram of the skin-score image with Otsu's threshold: H+S (two humps) vs Cr alone (no gap)."""
    skin, face, ign = _hand_masks(gt, frame.shape)
    fm = cv2.erode(face.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool); fm[:30] = False
    spaces = {"HSV (H,S)": "HSV (H,S)", "Cr alone": None}
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.5))
    res = {}
    for ax, name in zip(axes, spaces):
        if spaces[name]:
            h = _skin_model([frame], [fm], spaces[name]); bp = _cs_backproject(frame, h, spaces[name])
        else:
            cr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[..., 1].astype(int) // 8
            hc = np.bincount(cr[fm], minlength=32).astype(float)
            from scipy.ndimage import gaussian_filter1d
            hc = gaussian_filter1d(hc / hc.sum(), 1.0); bp = hc[cr]
        bp8 = cv2.GaussianBlur((255 * bp / max(bp.max(), 1e-12)).astype(np.uint8), (7, 7), 0)
        t, m = cv2.threshold(bp8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        gtm, valid = skin & ~face, ~ign & ~face
        hs_ = np.bincount(bp8[gtm & valid], minlength=256) / bp8[valid].size
        hn_ = np.bincount(bp8[~gtm & valid], minlength=256) / bp8[valid].size
        ax.bar(np.arange(256), hn_, 1.0, color="#7f8c8d", alpha=0.6, linewidth=0, label="not skin")
        ax.bar(np.arange(256), hs_, 1.0, color="#e67e22", alpha=0.7, linewidth=0, label="skin (hands, arms, neck)")
        ax.axvline(t, color="#c0392b", lw=1.5); ax.text(t + 3, 0.9, f"Otsu t* = {t:.0f}", color="#c0392b", fontsize=8, transform=ax.get_xaxis_transform())
        ax.set_yscale("log"); ax.set_xlim(0, 255)
        pred = m > 0                                                  # same scoring as `handtest`
        res[name] = (float(t), float((pred & gtm & valid).sum() / max(((pred | gtm) & valid).sum(), 1)))
        ax.set_title(f"{name}: skin-score histogram, mask IoU {res[name][1]:.2f}", fontsize=9)
        ax.set_xlabel("skin score (0–255)", fontsize=8); ax.tick_params(labelsize=7)
    axes[0].set_ylabel("share of pixels (log)", fontsize=8); axes[0].legend(fontsize=7, frameon=False, loc="upper right")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    return res


# ==============================================================================
# OVERVIEW: ADVANTAGES AND LIMITATIONS, TESTED
# Database matching (Part 5 slide), scale, efficiency, illumination, noise, rotation, shuffled pixels;
# also runs the Part 2 segmentation test. `analysis` command.
# ==============================================================================


def _crop_feature(img, box, space, grid, size=96, blur="gaussian"):
    """Normalized grid-histogram feature of one crop (resized to size x size)."""
    x, y, w, h = box
    c = cv2.resize(img[y:y + h, x:x + w], (size, size), interpolation=cv2.INTER_AREA)
    ch, bins, maxv = channel_spec(space)
    cp = preprocess(c, space, blur)
    ih = IntegralHistogram(cp, ch, bins, maxv, circular_channels(space), achrom=achromatic(cp, space))
    return window_feature(ih, 0, 0, size, size, grid), grid * grid * len(ch)


def _rotate(img, deg):
    """Rotate about the center on an expanded canvas (white border); returns image and 2x3 matrix."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    c, s_ = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * s_ + w * c), int(h * c + w * s_)
    M[0, 2] += nw / 2 - w / 2; M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh), borderValue=(255, 255, 255)), M


def run_analysis(photo_path, sheet_path):
    import json, time
    od = os.path.join(OUT, "analysis"); os.makedirs(od, exist_ok=True)
    photo, sheet = cv2.imread(photo_path), cv2.imread(sheet_path)
    tmpl, _ = stop_sign(120)
    R, lines = {}, []
    log = lambda t="": (print(t, flush=True), lines.append(t))
    best = lambda img, gt, **kw: iou(_best_window(detect(img, tmpl, **kw)[1])[1], gt)

    # 1. Step 5: classify the photo's stop sign against a database of the 18 sheet signs
    log("1) DATABASE MATCHING: query = stop sign cropped from the photo; database = 18 signs from the sign sheet")
    R["database"] = {}
    for sp in ["gray", "rgb", "hsv", "lab"]:
        for grid in [1, 3]:
            db = {n: _crop_feature(sheet, b, sp, grid)[0] for n, b in SHEET_SIGNS.items()}
            q, nh = _crop_feature(photo, PHOTO_GT, sp, grid)
            for m in ["d1", "d2", "d3"]:
                d = {n: float(METRICS[m](q, f) / nh) for n, f in db.items()}
                order = sorted(d, key=d.get)
                R["database"][f"{sp}-g{grid}-{m}"] = dict(order=order[:3], d=[d[o] for o in order[:3]], stop_rank=order.index("STOP") + 1)
                log(f"   {sp:5} grid {grid} {m}: predicted {order[0]:14} STOP rank {order.index('STOP') + 1:2d}   top-3 " +
                    ", ".join(f"{o} {d[o]:.3f}" for o in order[:3]))

    # 2. multi-scale: best D3 at each window size (HSV), at the sign vs anywhere else
    log("\n2) MULTI-SCALE (HSV, D3): best distance per window size, on the sign vs elsewhere")
    _, maps, thr = detect(photo, tmpl, space="hsv", metric="d3")
    sizes, on, off = [], [], []
    for hh in sorted(maps):
        bs, ds = maps[hh]
        ov = np.array([iou(b, PHOTO_GT) for b in bs])
        sizes.append(hh); on.append(float(ds[ov > 0.3].min()) if (ov > 0.3).any() else np.nan)
        off.append(float(ds[ov == 0].min()))
    R["scale"] = dict(sizes=sizes, on_sign=on, elsewhere=off, threshold=float(thr))
    for a, b_, c in zip(sizes, on, off):
        log(f"   window {a:4d} px: on sign {b_:.3f}  elsewhere {c:.3f}")
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(sizes, on, "o-", color="#c0392b", label="best window on the sign (IoU > 0.3)")
    ax.plot(sizes, off, "s-", color="#7f8c8d", label="best window elsewhere")
    ax.axhline(thr, ls="--", color="#2c3e50", lw=1, label=f"threshold (mean − 2.5σ) = {thr:.2f}")
    ax.axvline(PHOTO_GT[3], ls=":", color="#c0392b", lw=1); ax.text(PHOTO_GT[3] + 3, max(off) * 0.97, "sign = 66 px", color="#c0392b", fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("window size (px, log scale)"); ax.set_ylabel("D3 (lower = more similar)")
    ax.set_xticks([25, 40, 66, 100, 150, 200], ["25", "40", "66", "100", "150", "200"]); ax.minorticks_off()
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    plt.tight_layout(); plt.savefig(f"{od}/scale_curve.png", dpi=150); plt.close()

    # 3. shuffled pixels: same histogram, different image
    log("\n3) SHUFFLED PIXELS: photo sign window vs the same pixels shuffled (HSV, D3, no blur)")
    rng = np.random.default_rng(0)
    x, y, w, h = PHOTO_GT
    crop = photo[y:y + h, x:x + w].copy()
    shuf = crop.reshape(-1, 3)[rng.permutation(w * h)].reshape(crop.shape)
    R["shuffle"] = {}
    for grid in [1, 3]:
        fa, nh = _crop_feature(crop, (0, 0, w, h), "hsv", grid, size=w, blur="none")   # raw histogram: no blur
        fb, _ = _crop_feature(shuf, (0, 0, w, h), "hsv", grid, size=w, blur="none")
        R["shuffle"][grid] = float(d3(fa, fb) / nh)
        log(f"   grid {grid}x{grid}: D3(original, shuffled) = {R['shuffle'][grid]:.4f}")
    fig, ax = plt.subplots(1, 2, figsize=(4.4, 2.4))
    for a_, im, t in zip(ax, [crop, shuf], ["sign window", "same pixels, shuffled"]):
        a_.imshow(cv2.cvtColor(cv2.resize(im, (198, 198), interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2RGB)); a_.set_title(t, fontsize=10); a_.axis("off")
    plt.tight_layout(); plt.savefig(f"{od}/shuffle.png", dpi=150); plt.close()

    # 4. efficiency
    log("\n4) EFFICIENCY: full multi-scale scan of the photo (HSV, D3)")
    t0 = time.perf_counter(); _, maps, _ = detect(photo, tmpl, space="hsv", metric="d3"); dt = time.perf_counter() - t0
    nwin = sum(len(bs) for bs, _ in maps.values())
    R["timing"] = dict(seconds=dt, windows=nwin, sizes=len(maps))
    log(f"   {nwin} windows at {len(maps)} sizes in {dt:.2f} s ({1e6 * dt / nwin:.0f} µs per window, 3x3 grid, 27 bins per cell)")

    # 5. illumination
    log("\n5) ILLUMINATION: best-window IoU on the relit photo (D3)")
    R["light"] = {}
    for sp in ["gray", "rgb", "hsv", "lab"]:
        row = [best(_relight(photo, *L), PHOTO_GT, space=sp, metric="d3") for L in CS_LIGHTS.values()]
        R["light"][sp] = row
        log(f"   {sp:5} " + "  ".join(f"{k} {v:.2f}" for k, v in zip(CS_LIGHTS, row)))

    # 6. noise x noise reduction (step 1B)
    log("\n6) NOISE: Gaussian noise added, HSV D3, three noise-reduction settings")
    R["noise"] = {}
    for blur in ["none", "gaussian", "median"]:
        row = []
        for sig in [0, 10, 20, 40]:
            noisy = np.clip(photo.astype(float) + np.random.default_rng(1).normal(0, sig, photo.shape), 0, 255).astype(np.uint8)
            row.append(best(noisy, PHOTO_GT, space="hsv", metric="d3", blur=blur))
        R["noise"][blur] = row
        log(f"   blur {blur:8} " + "  ".join(f"σ={s_} {v:.2f}" for s_, v in zip([0, 10, 20, 40], row)))

    # 7. rotation
    log("\n7) ROTATION: rotated photo, unrotated template (HSV, D3)")
    R["rotation"] = {}
    cx, cy = x + w / 2, y + h / 2
    for deg in [0, 15, 30, 45, 90]:
        img, M = _rotate(photo, deg)
        nx, ny = M @ np.array([cx, cy, 1.0])
        gt = (int(nx - w / 2), int(ny - h / 2), w, h)
        row = {g: best(img, gt, space="hsv", metric="d3", grid=g) for g in [1, 3]}
        R["rotation"][deg] = row
        log(f"   {deg:3d}°: grid 1x1 IoU {row[1]:.2f}   grid 3x3 IoU {row[3]:.2f}")

    run_segmentation(photo, sheet, tmpl, od, log, R)
    json.dump(R, open(f"{od}/analysis.json", "w"), indent=1, default=float)
    open(f"{od}/analysis_results.txt", "w").write("\n".join(lines) + "\n")


# ==============================================================================
# COMMAND LINE
# ==============================================================================


# -------------------- `histograms` command

def run_histograms(photo_path="stop_photo.png", sheet_path="sign_sheet.png", frame_path="asl_f1.png"):
    """Histogram figures for the slides (Parts 1A, 1B, 3, 5 and the hand application)."""
    od = os.path.join(OUT, "histograms"); os.makedirs(od, exist_ok=True)
    photo, sheet, f1 = cv2.imread(photo_path), cv2.imread(sheet_path), cv2.imread(frame_path)
    if photo is None or sheet is None or f1 is None:
        raise SystemExit("could not read the photo, the sign sheet or the ASL frame")
    tmpl, _ = stop_sign(120)
    lines = []
    log = lambda t="": (print(t), lines.append(t))
    r = fig_color_hists(photo, sheet, tmpl, f"{od}/hist_colors.png")
    log("Part 1A, whole-sign D3 to the template (1 cell):")
    for sp, (dp, ds) in r.items():
        log(f"   {sp:4}  photo sign {dp:.3f}   sheet STOP {ds:.3f}")
    r = fig_noise_hists(photo, tmpl, f"{od}/hist_noise.png")
    log("Part 1B, center cell of the photo sign, D3 to the template's center cell:")
    for k, v in r.items():
        log(f"   {k:28} {v:.3f}")
    fig_hue_fixes(photo, f"{od}/hist_hue_fixes.png")
    r = fig_sign_hists(sheet, tmpl, f"{od}/hist_red_signs.png")
    log("Part 5, sign sheet, D3 to the template (1 cell, 3 x 3):")
    for k, (a, b) in r.items():
        log(f"   {k:14} {a:.3f}  {b:.3f}")
    r = fig_skin_channel_hists(f1, HAND_GT["f1"], f"{od}/hist_skin_channels.png")
    log("Hands (f1), overlap of skin and non-skin histograms (0 = separate, 1 = identical):")
    log("   " + "  ".join(f"{k} {v:.2f}" for k, v in r.items()))
    r = fig_otsu_hists(f1, HAND_GT["f1"], f"{od}/hist_otsu.png")
    log("Hands (f1), Otsu threshold and mask IoU:")
    for k, (t, m) in r.items():
        log(f"   {k:10} t* = {t:.0f}   mask IoU {m:.2f}")
    open(f"{od}/histograms_results.txt", "w").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Local-histogram object detector")
    sub = ap.add_subparsers(dest="cmd")
    # sub.add_parser("demo", help="synthetic scene + evaluation grid + figures")  # disabled
    sub.add_parser("features", help="slide-6 histogram features (~3 min)")
    sub.add_parser("histograms", help="histogram figures for the slides (~30 s)")
    sub.add_parser("colorspace", help="Gray, RGB, HSV, Lab, YCrCb: stop sign vs. skin/hands")
    pr = sub.add_parser("realtest", help="street photo + sign sheet tests")
    pr.add_argument("--photo", default="stop_photo.png"); pr.add_argument("--sheet", default="sign_sheet.png")
    ph = sub.add_parser("handtest", help="skin/hand segmentation on 3 ASL frames")
    ph.add_argument("frames", nargs=3, help="f1 f2 f3 (How2Sign, How2Sign, YouTube)")
    pa = sub.add_parser("analysis", help="step-5 database matching + tested advantages/limitations")
    pa.add_argument("--photo", default="stop_photo.png"); pa.add_argument("--sheet", default="sign_sheet.png")
    sub.add_parser("all", help="features + colorspace (default)")
    p = sub.add_parser("detect", help="run detector on your own images")
    p.add_argument("scene"); p.add_argument("template")
    p.add_argument("-o", "--out", default="detections.png")
    p.add_argument("--space", default="hsv", choices=SPACES)
    p.add_argument("--metric", default="d3", choices=METRICS)
    p.add_argument("--grid", type=int, default=3)
    p.add_argument("--blur", default="gaussian", choices=["gaussian", "median", "none"])
    p.add_argument("--ksize", type=int, default=5)
    p.add_argument("--stride", type=float, default=0.1)
    p.add_argument("--thresh", type=float, default=None)
    p.add_argument("--use-v", action="store_true")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--stats-w", type=float, default=0.0, help="weight of slide-4 statistics: mean, alpha-trimmed mean, median, mode, std (0 = off)")
    a = ap.parse_args()

    if a.cmd in (None, "all"):
        run_features(); run_colorspace()  # run_demo() disabled
    elif a.cmd == "handtest":
        run_handtest(a.frames)
    elif a.cmd == "analysis":
        run_analysis(a.photo, a.sheet)
    elif a.cmd == "realtest":
        run_realtest(a.photo, a.sheet)
    elif a.cmd == "colorspace":
        run_colorspace()
    elif a.cmd == "histograms":
        run_histograms()
    elif a.cmd == "features":
        run_features()
    else:
        run_detect(a)




# ==============================================================================
# DISABLED: synthetic-scene demo (its outputs aren't used in the slides)
# ==============================================================================

# def run_demo():
#     os.makedirs(OUT, exist_ok=True)
#     scene, gt = scene_img()
#     tmpl, tmask = stop_sign(120)
#     cv2.imwrite(f"{OUT}/scene.png", scene)
#     cv2.imwrite(f"{OUT}/template.png", np.dstack([tmpl, tmask.astype(np.uint8) * 255]))  # RGBA
#
#     # --- evaluation grid: color space x metric x template mask
#     rows = []
#     for use_mask in [False, True]:
#         for space in ["gray", "rgb", "hsv", "lab"]:
#             for m in METRICS:
#                 dets, _, thr = detect(scene, tmpl, space=space, metric=m,
#                                       template_mask=tmask if use_mask else None)
#                 top = dets[0] if dets else (np.nan, (0, 0, 1, 1))
#                 bx, by, bw, bh = top[1]; cx, cy = bx + bw / 2, by + bh / 2
#                 on = gt[0] <= cx <= gt[0] + gt[2] and gt[1] <= cy <= gt[1] + gt[3]
#                 rows.append((use_mask, space, m, top[0], iou(top[1], gt), on))
#     print(f"ground truth box: {gt}")
#     hdr = f"{'mask':5} {'space':6} {'metric':6} {'top-1 d':>8} {'IoU':>6}  on-sign  IoU>=0.5"
#     lines = [hdr] + [f"{str(r[0]):5} {r[1]:6} {r[2]:6} {r[3]:8.4f} {r[4]:6.2f}  {'Y' if r[5] else 'n':^7}  {'Y' if r[4] >= 0.5 else 'n':^8}" for r in rows]
#     print("\n".join(lines)); open(f"{OUT}/results.txt", "w").write("\n".join(lines) + "\n")
#
#     # --- main result (HSV, D3, masked template)
#     dets, maps, thr = detect(scene, tmpl, space="hsv", metric="d3", template_mask=tmask)
#     vis = draw(scene, dets)
#     x, y, w, h = gt
#     cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 255), 1)
#     cv2.imwrite(f"{OUT}/detections_hsv_d3.png", vis)
#
#     # similarity heatmap at the best scale
#     best_h = dets[0][1][3]
#     boxes, dists = maps[best_h]
#     heat = np.full(scene.shape[:2], np.nan)
#     for (bx, by, bw, bh), d in zip(boxes, dists):
#         cy, cx = by + bh // 2, bx + bw // 2
#         heat[cy, cx] = d
#     pts = np.array([(b[1] + b[3] // 2, b[0] + b[2] // 2) for b in boxes])
#     fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
#     ax[0].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); ax[0].set_title("Detections (HSV, D3) — magenta = ground truth"); ax[0].axis("off")
#     sc = ax[1].scatter(pts[:, 1], pts[:, 0], c=-dists, s=14, cmap="magma", marker="s")
#     ax[1].set_xlim(0, scene.shape[1]); ax[1].set_ylim(scene.shape[0], 0); ax[1].set_aspect("equal")
#     ax[1].set_title(f"Similarity map (−D3) at window size {best_h}px"); plt.colorbar(sc, ax=ax[1], fraction=0.03)
#     plt.tight_layout(); plt.savefig(f"{OUT}/heatmap.png", dpi=110); plt.close()
#
#     # --- slide-style figure: scene blocks -> histograms -> target
#     ch, bins, maxv = channel_spec("hsv")
#     tp, sp = preprocess(tmpl), preprocess(scene)
#     s_ih = IntegralHistogram(sp, ch, bins, maxv, (0,), achrom=achromatic(sp, "hsv"))
#     t_ih = IntegralHistogram(tp, ch, bins, maxv, (0,), tmask, achromatic(tp, "hsv"))
#     tf = window_feature(t_ih, 0, 0, 120, 120, 1)
#     blocks = {"stop sign": gt, "red roof": (480, 70, 58, 58), "mailbox": (50, 290, 58, 58),
#               "road": (300, 340, 58, 58), "tree": (100, 60, 58, 58)}
#     fig, axes = plt.subplots(2, len(blocks) + 1, figsize=(15, 5))
#     for k, (name, box) in enumerate(list(blocks.items()) + [("TARGET", None)]):
#         if name == "TARGET":
#             patch, f = tmpl, tf
#         else:
#             bx, by, bw, bh = box
#             patch, f = scene[by:by + bh, bx:bx + bw], window_feature(s_ih, by, bx, bh, bw, 1)
#         axes[0, k].imshow(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)); axes[0, k].axis("off")
#         title = name if name == "TARGET" else f"{name}\nD1={METRICS['d1'](f, tf)/2:.2f}  D3={METRICS['d3'](f, tf)/2:.2f}"
#         axes[0, k].set_title(title, fontsize=9)
#         axes[1, k].bar(range(len(f)), f, color=["#c0392b"] * bins[0] + ["#999999"] + ["#2c3e50"] * bins[1])
#         axes[1, k].set_xticks([bins[0] / 2, bins[0], bins[0] + 1 + bins[1] / 2], ["Hue", "gray", "Sat"]); axes[1, k].set_yticks([])
#     plt.suptitle("Local HSV histograms vs. target (lower distance = more similar)")
#     plt.tight_layout(); plt.savefig(f"{OUT}/histogram_comparison.png", dpi=110); plt.close()
#
#     # --- descriptive statistics for the best detection
#     bx, by, bw, bh = dets[0][1]
#     stats = descriptive_stats(preprocess(scene, "hsv")[by:by + bh, bx:bx + bw])
#     print("\nDescriptive stats of top detection (HSV):")
#     for c, name in zip(range(3), "HSV"):
#         print(name, {k: round(float(v), 1) for k, v in stats[c].items()})
#


if __name__ == "__main__":
    main()
