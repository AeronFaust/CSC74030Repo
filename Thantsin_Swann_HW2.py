# __define-ocg__
# Logarithmic Image Processing (LIP) - add, subtract, multiply, divide
# Usage:
#   python lip_simple.py frames/            -> pairs consecutive frames (1&2, 2&3, ...)
#   python lip_simple.py gt/ recon/         -> pairs files with the same name

import sys
import os
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt

M = 256.0  # LIP upper bound (8-bit images)


# ---------- 1. Convert between intensity and LIP gray tone ----------
# LIP works on "gray tones": 0 = white, M = black (like light absorbed)
def to_tone(img):
    return M - img.astype(np.float64)

def to_image(tone):
    return np.clip(M - tone, 0, 255).astype(np.uint8)


# ---------- 2. Isomorphism: maps LIP space <-> real numbers ----------
# Lets us do multiply/divide as normal math, then map back
def phi(f):
    return -M * np.log(np.clip(1 - f / M, 1e-6, None))

def phi_inv(x):
    return M * (1 - np.exp(-x / M))

UNIT = phi(M / 2)  # mid-gray acts as "1" for multiply/divide


# ---------- 3. The four LIP operations ----------
def lip_add(f, g):
    # never exceeds M, so no saturation
    return f + g - f * g / M

def lip_sub(f, g):
    # difference scaled by brightness of g (matches human vision)
    return M * (f - g) / np.clip(M - g, 1e-6, None)

def lip_mul(f, g):
    return phi_inv(phi(f) * phi(g) / UNIT)

def lip_div(f, g):
    return phi_inv(UNIT * phi(f) / np.clip(phi(g), 1e-6, None))


# ---------- 4. Classical (regular) arithmetic for comparison ----------
def cls_add(a, b): return np.clip(a.astype(float) + b, 0, 255).astype(np.uint8)
def cls_sub(a, b): return np.abs(a.astype(float) - b).astype(np.uint8)  # difference map
def cls_mul(a, b): return np.clip(a.astype(float) * b / 255, 0, 255).astype(np.uint8)
def cls_div(a, b): return np.clip(255 * a.astype(float) / np.clip(b.astype(float), 1, None), 0, 255).astype(np.uint8)


# ---------- 5. Simple quality measures ----------
def entropy(img):
    # how much information/detail the image holds (higher = more)
    p = np.bincount(img.ravel(), minlength=256) / img.size
    p = p[p > 0]
    return -(p * np.log2(p)).sum()

def saturated(img):
    # % of pixels stuck at pure black or white (lost detail)
    return 100 * np.mean((img == 0) | (img == 255))


# ---------- 6. Process one pair of images ----------
def process_pair(a, b, name, out_dir):
    b = cv2.resize(b, (a.shape[1], a.shape[0]))  # match sizes
    fa, fb = to_tone(a), to_tone(b)

    # apply every operation (classical vs LIP)
    results = {
        "Classical add": cls_add(a, b),
        "LIP add":       to_image(lip_add(fa, fb)),
        "Classical sub": cls_sub(a, b),
        "LIP sub":       np.clip(np.abs(lip_sub(fa, fb)), 0, 255).astype(np.uint8),  # difference map
        "Classical mul": cls_mul(a, b),
        "LIP mul":       to_image(lip_mul(fa, fb)),
        "Classical div": cls_div(a, b),
        "LIP div":       to_image(lip_div(fa, fb)),
    }

    # save comparison grid for this pair
    panels = {"Input A": a, "Input B": b, **results}
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for ax, (title, img) in zip(axes.ravel(), panels.items()):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}.png"), dpi=100)
    plt.close(fig)

    # return metrics for this pair
    return [{"pair": name, "operation": op,
             "entropy": round(entropy(img), 3),
             "saturated_pct": round(saturated(img), 2)}
            for op, img in results.items()]


# ---------- 7. Build list of image pairs from folder(s) ----------
EXTS = (".png", ".jpg", ".jpeg", ".bmp")

def list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(EXTS))

if len(sys.argv) == 2:
    # one folder: consecutive frames
    folder = sys.argv[1]
    files = list_images(folder)
    pairs = [(os.path.join(folder, files[i]), os.path.join(folder, files[i + 1]))
             for i in range(len(files) - 1)]
else:
    # two folders: same filename in each
    fa_dir, fb_dir = sys.argv[1], sys.argv[2]
    common = sorted(set(list_images(fa_dir)) & set(list_images(fb_dir)))
    pairs = [(os.path.join(fa_dir, f), os.path.join(fb_dir, f)) for f in common]

print(f"Found {len(pairs)} image pairs")


# ---------- 8. Run every pair, save figures + metrics ----------
out_dir = "lip_results"
os.makedirs(out_dir, exist_ok=True)
rows = []
for pa, pb in pairs:
    a = cv2.imread(pa, cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(pb, cv2.IMREAD_GRAYSCALE)
    name = os.path.splitext(os.path.basename(pa))[0]
    rows += process_pair(a, b, name, out_dir)
    print("done:", name)

with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["pair", "operation", "entropy", "saturated_pct"])
    writer.writeheader()
    writer.writerows(rows)


# ---------- 9. Print average metrics per operation ----------
print(f"\n{'Operation':15s} {'Avg entropy':>12s} {'Avg saturated %':>16s}")
for op in dict.fromkeys(r["operation"] for r in rows):
    sel = [r for r in rows if r["operation"] == op]
    print(f"{op:15s} {np.mean([r['entropy'] for r in sel]):12.2f} "
          f"{np.mean([r['saturated_pct'] for r in sel]):16.1f}")
