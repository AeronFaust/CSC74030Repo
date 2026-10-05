# __define-ocg__
# Color pre-correction (white balance / color constancy) and color preservation
# Usage: python color_correction.py images/

import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

SIMULATE_CAST = True               # add a lighting color cast to test white balance
CAST = (1.00, 0.80, 0.55)          # R, G, B gains: warm indoor (tungsten-like) light
PCT = 99                           # white patch uses the 99th percentile, not the max
EPS = 1e-6


# ---------- 1. Load RGB in [0, 1] ----------
def load(path):
    img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return img.astype(np.float64) / 255.0


# ---------- 2. Simulate a color cast (test input) ----------
# the camera sees surface color x light color, channel by channel
def add_cast(rgb, gains=CAST):
    return np.clip(rgb * np.array(gains), 0, 1)


# ---------- 3. Pre-correction: Gray World ----------
# assumption: the average scene color is gray, so equalize channel means
def gray_world(rgb):
    means = rgb.reshape(-1, 3).mean(axis=0)
    gains = means.mean() / (means + EPS)
    return np.clip(rgb * gains, 0, 1)


# ---------- 4. Pre-correction: White Patch (max-RGB) ----------
# assumption: the brightest surface is white, so scale it to (1, 1, 1)
def white_patch(rgb, pct=PCT):
    brightest = np.percentile(rgb.reshape(-1, 3), pct, axis=0)  # robust to a few hot pixels
    return np.clip(rgb / (brightest + EPS), 0, 1)


# ---------- 5. Enhancement WITHOUT color preservation ----------
# histogram-equalize R, G, B separately: ratios between channels change
def equalize_channels(rgb):
    u8 = (rgb * 255).astype(np.uint8)
    out = np.stack([cv2.equalizeHist(u8[..., c]) for c in range(3)], axis=-1)
    return out.astype(np.float64) / 255.0


# ---------- 6. Enhancement WITH color preservation ----------
# equalize intensity only, then scale R, G, B by the same ratio I'/I,
# so R:G:B (hue and saturation) stay the same
def equalize_preserving(rgb):
    I = rgb.mean(axis=-1)                                     # intensity
    I_eq = cv2.equalizeHist((I * 255).astype(np.uint8)) / 255.0
    ratio = I_eq / (I + EPS)
    ratio = np.minimum(ratio, 1.0 / (rgb.max(axis=-1) + EPS)) # cap: no channel above 1 (gamut)
    return np.clip(rgb * ratio[..., None], 0, 1)


# ---------- 7. Process one image, run folder ----------
def save_row(panels, path):
    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.8))
    for ax, (title, im) in zip(axes, panels.items()):
        ax.imshow(np.clip(im, 0, 1))
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=80)
    plt.close(fig)

def process(path, out_dir):
    original = load(path)
    name = os.path.splitext(os.path.basename(path))[0]
    cast = add_cast(original) if SIMULATE_CAST else original

    # pre-correction (color constancy)
    gw, wp = gray_world(cast), white_patch(cast)
    save_row({"Original": original, "Color cast input": cast,
              "Gray World": gw, "White Patch": wp},
             os.path.join(out_dir, f"{name}_whitebalance.png"))

    # color preservation, applied after pre-correction
    save_row({"Pre-corrected (Gray World)": gw,
              "Per-channel equalization": equalize_channels(gw),
              "Color-preserving equalization": equalize_preserving(gw)},
             os.path.join(out_dir, f"{name}_preservation.png"))


if __name__ == "__main__":
    folder = sys.argv[1]
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(exts))
    out_dir = "correction_results"
    os.makedirs(out_dir, exist_ok=True)
    for f in files:
        process(os.path.join(folder, f), out_dir)
        print("done:", f)
