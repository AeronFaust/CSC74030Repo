# __define-ocg__
# Retinex decomposition: image I = reflectance R x illumination L
# Single-Scale (SSR) and Multi-Scale (MSR) Retinex + low-light enhancement
# Usage: python retinex.py images/

import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

SIGMA  = 80              # blur size for SSR illumination
SIGMAS = (15, 80, 250)   # small / medium / large scales for MSR
GAMMA  = 0.3             # illumination brightening (< 1 brightens)
EPS    = 1e-4            # avoids log(0) and divide by 0


# ---------- 1. Load image as float in [0, 1] ----------
def load(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float64) / 255.0 + EPS


# ---------- 2. Estimate illumination L ----------
# lighting changes slowly across a scene, so a large blur keeps
# the lighting and removes object detail
def illumination(img, sigma):
    brightest = img.max(axis=2)                  # brightest channel per pixel
    L = cv2.GaussianBlur(brightest, (0, 0), sigma)
    return np.clip(L, EPS, 1.0)


# ---------- 3. Single-Scale Retinex (SSR) ----------
# I = R * L  ->  log R = log I - log L
def ssr(img, sigma=SIGMA):
    L = illumination(img, sigma)
    log_R = np.log(img) - np.log(L)[..., None]
    return log_R, L


# ---------- 4. Multi-Scale Retinex (MSR) ----------
# average SSR over several scales: small = detail, large = color/tone
def msr(img, sigmas=SIGMAS):
    return np.mean([ssr(img, s)[0] for s in sigmas], axis=0)


# ---------- 5. Low-light enhancement ----------
# keep reflectance, brighten only the illumination: I' = R * L^gamma
def enhance(img, sigma=SIGMA, gamma=GAMMA):
    log_R, L = ssr(img, sigma)
    R = np.clip(np.exp(log_R), 0, 1)
    return R * (L ** gamma)[..., None]


# ---------- 6. Display helpers ----------
def stretch(x):
    # map values to 0-255 using 1st-99th percentile (log R has no fixed range)
    lo, hi = np.percentile(x, (1, 99))
    return (np.clip((x - lo) / (hi - lo + EPS), 0, 1) * 255).astype(np.uint8)

def to_uint8(x):
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


# ---------- 7. Process one image ----------
def process(path, out_dir):
    img = load(path)
    log_R, L = ssr(img)

    results = {
        "Input":               to_uint8(img),
        "Illumination L":      stretch(L),  # contrast-stretched to be visible
        "Reflectance R (SSR)": stretch(log_R),
        "Reflectance R (MSR)": stretch(msr(img)),
        "Enhanced R·L^γ":      to_uint8(enhance(img)),
    }

    # save one row: input, decomposition, enhancement
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    for ax, (title, im) in zip(axes, results.items()):
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    name = os.path.splitext(os.path.basename(path))[0]
    plt.savefig(os.path.join(out_dir, f"{name}.png"), dpi=90)
    plt.close(fig)

    # also save each output separately for slides
    for title, im in results.items():
        slug = "".join(ch if ch.isalnum() else "_" for ch in title).strip("_")
        bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR) if im.ndim == 3 else im
        cv2.imwrite(os.path.join(out_dir, f"{name}__{slug}.png"), bgr)


# ---------- 8. Run on every image in the folder ----------
if __name__ == "__main__":
    folder = sys.argv[1]
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(exts))
    print(f"Found {len(files)} images")

    out_dir = "retinex_results"
    os.makedirs(out_dir, exist_ok=True)
    for f in files:
        process(os.path.join(folder, f), out_dir)
        print("done:", f)
