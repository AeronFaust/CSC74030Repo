# __define-ocg__
# Demosaicking: rebuild full color from a Bayer mosaic (one color per pixel)
#   baseline: bilinear interpolation
#   main:     Hamilton-Adams (gradient-directed green + color-difference R, B)
# Usage: python demosaic.py images/

import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

CROP = 96     # size of the zoomed detail window
ZOOM = 4      # zoom factor for the detail window


# ---------- 1. Load RGB in [0, 1] ----------
def load(path):
    img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return img.astype(np.float64) / 255.0


# ---------- 2. Bayer masks (RGGB, as on the lecture slide) ----------
# row 0: R G R G ...   row 1: G B G B ...
def bayer_masks(h, w):
    y, x = np.mgrid[:h, :w]
    r = (y % 2 == 0) & (x % 2 == 0)
    b = (y % 2 == 1) & (x % 2 == 1)
    g = ~(r | b)
    return r, g, b


# ---------- 3. Simulate the sensor: keep one color per pixel ----------
def mosaic(rgb):
    r, g, b = bayer_masks(*rgb.shape[:2])
    cfa = rgb[..., 0] * r + rgb[..., 1] * g + rgb[..., 2] * b
    return cfa, (r, g, b)


def show_mosaic(cfa, masks):
    # color each sample by its filter, for display only
    return np.stack([cfa * m for m in masks], axis=-1)


# ---------- 4. Baseline: bilinear interpolation ----------
K_G  = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]]) / 4    # average of 4 neighbors
K_RB = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 4    # 2 or 4 neighbors

def conv(x, k):
    return cv2.filter2D(x, -1, k, borderType=cv2.BORDER_REFLECT)

def bilinear(cfa, masks):
    r, g, b = masks
    R = conv(cfa * r, K_RB)
    G = conv(cfa * g, K_G)
    B = conv(cfa * b, K_RB)
    return np.clip(np.stack([R, G, B], axis=-1), 0, 1)


# ---------- 5. Hamilton-Adams: green at red and blue pixels ----------
# X = the red or blue value at the center (R5 on the slide)
def ha_green(cfa, masks):
    r, g, b = masks
    p = np.pad(cfa, 2, mode="reflect")
    c = lambda dy, dx: p[2 + dy: p.shape[0] - 2 + dy, 2 + dx: p.shape[1] - 2 + dx]

    X5 = cfa
    G2, G8, G4, G6 = c(-1, 0), c(1, 0), c(0, -1), c(0, 1)   # green neighbors
    X1, X9, X3, X7 = c(-2, 0), c(2, 0), c(0, -2), c(0, 2)   # same color, 2 away

    # step 1-2: gradients (green difference + second derivative of center color)
    dH = np.abs(G4 - G6) + np.abs(2 * X5 - X3 - X7)
    dV = np.abs(G2 - G8) + np.abs(2 * X5 - X1 - X9)

    # step 3: interpolate along the smoother direction
    g_vert = (G2 + G8) / 2 + (2 * X5 - X1 - X9) / 4
    g_horz = (G4 + G6) / 2 + (2 * X5 - X3 - X7) / 4
    g_both = (G2 + G8 + G4 + G6) / 4 + (4 * X5 - X1 - X9 - X3 - X7) / 8
    G_est = np.where(dH > dV, g_vert, np.where(dH < dV, g_horz, g_both))

    return np.clip(np.where(g, cfa, G_est), 0, 1)           # keep measured greens


# ---------- 6. Hamilton-Adams: red and blue from color differences ----------
# R - G varies smoothly, so interpolate the difference, then add G back
def ha_rb(cfa, G, mask):
    diff = (cfa - G) * mask
    return np.clip(G + conv(diff, K_RB), 0, 1)

def hamilton_adams(cfa, masks):
    r, g, b = masks
    G = ha_green(cfa, masks)
    return np.stack([ha_rb(cfa, G, r), G, ha_rb(cfa, G, b)], axis=-1)


# ---------- 7. Find a detailed region to zoom into ----------
def detail_window(rgb, size=CROP):
    gray = rgb.mean(axis=-1)
    edges = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0)) + np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    score = cv2.boxFilter(edges, -1, (size, size))
    h, w = gray.shape
    score[: size // 2, :] = score[h - size // 2:, :] = 0      # keep window inside image
    score[:, : size // 2] = score[:, w - size // 2:] = 0
    cy, cx = np.unravel_index(score.argmax(), score.shape)
    return slice(cy - size // 2, cy + size // 2), slice(cx - size // 2, cx + size // 2)


# ---------- 8. Process one image, run folder ----------
def process(path, out_dir):
    rgb = load(path)
    cfa, masks = mosaic(rgb)
    panels = {"Original": rgb,
              "Bayer mosaic (sensor)": show_mosaic(cfa, masks),
              "Bilinear": bilinear(cfa, masks),
              "Hamilton-Adams": hamilton_adams(cfa, masks)}

    ys, xs = detail_window(rgb)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9.5))
    for c, (title, im) in enumerate(panels.items()):
        axes[0, c].imshow(im)
        axes[0, c].set_title(title)
        axes[0, c].add_patch(plt.Rectangle((xs.start, ys.start), CROP, CROP, fill=False, ec="white", lw=1.5))
        crop = np.kron(im[ys, xs], np.ones((ZOOM, ZOOM, 1)))  # nearest-neighbor zoom
        axes[1, c].imshow(crop)
        axes[1, c].set_title(f"{title}: detail x{ZOOM}")
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    name = os.path.splitext(os.path.basename(path))[0]
    plt.savefig(os.path.join(out_dir, f"{name}_demosaic.png"), dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    folder = sys.argv[1]
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(exts))
    out_dir = "demosaic_results"
    os.makedirs(out_dir, exist_ok=True)
    for f in files:
        process(os.path.join(folder, f), out_dir)
        print("done:", f)
