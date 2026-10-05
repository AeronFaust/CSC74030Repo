# __define-ocg__
# Grayscale -> RGB pseudo-color
#   (a) intensity slicing: k gray regions, one color each
#   (b) transfer functions: R, G, B each a function of f(x,y)
# Usage: python pseudocolor.py images/

import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

L = 255          # gray range 0 <= f(x,y) <= L
K_LIST = (4, 8)  # numbers of regions to show


# ---------- 1. Load as grayscale f(x,y) ----------
def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)
    return 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]


# ---------- 2. Transfer functions (lecture rules) ----------
# blue -> cyan -> green -> yellow -> red as f goes 0 -> 255
def transfer(f):
    f = np.asarray(f, dtype=np.float64)
    R = np.zeros_like(f); G = np.zeros_like(f); B = np.zeros_like(f)

    # integer ranges 0-63, 64-127, 128-191, 192-255 (written with < so decimals fit)
    m = f < 64                               # blue, green rising
    R[m], G[m], B[m] = 0, 4 * f[m], 255
    m = (f >= 64) & (f < 128)                # green, blue falling
    R[m], G[m], B[m] = 0, 255, 511 - 4 * f[m]
    m = (f >= 128) & (f < 192)               # green, red rising
    R[m], G[m], B[m] = 4 * f[m] - 511, 255, 0
    m = f >= 192                             # red, green falling
    R[m], G[m], B[m] = 255, 1023 - 4 * f[m], 0
    return np.clip(np.stack([R, G, B], axis=-1), 0, 255).astype(np.uint8)


# ---------- 3. Intensity slicing ----------
# k + 1 levels I0 = 0 ... Ik = L; region i gets color Ci
def slicing(f, k):
    levels = np.linspace(0, L, k + 1)                   # I0, I1, ..., Ik
    region = np.clip(np.digitize(f, levels[1:-1]), 0, k - 1)  # region index per pixel
    centers = (levels[:-1] + levels[1:]) / 2
    colors = transfer(centers)                          # Ci = color at region center
    return colors[region], levels, colors


# ---------- 4. Plot the transfer functions (for slides) ----------
def plot_transfer(out_dir):
    f = np.arange(0, 256)
    rgb = transfer(f)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, c, name in zip(axes, range(3), ["f_R", "f_G", "f_B"]):
        ax.plot(f, rgb[:, c], color="black")
        ax.set_title(name + "(x,y)")
        ax.set_xlabel("f(x,y)")
        ax.set_xlim(0, 255); ax.set_ylim(-5, 260)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "transfer_functions.png"), dpi=100)
    plt.close(fig)


# ---------- 5. Process one image, run folder ----------
def process(path, out_dir):
    f = load_gray(path)
    panels = {"Grayscale f(x,y)": f}
    for k in K_LIST:
        panels[f"Slicing, k = {k}"] = slicing(f, k)[0]
    panels["Transfer functions"] = transfer(f)

    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.8))
    for ax, (title, im) in zip(axes, panels.items()):
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    name = os.path.splitext(os.path.basename(path))[0]
    plt.savefig(os.path.join(out_dir, f"{name}_pseudo.png"), dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    folder = sys.argv[1]
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(exts))
    out_dir = "pseudo_results"
    os.makedirs(out_dir, exist_ok=True)
    plot_transfer(out_dir)
    for f in files:
        process(os.path.join(folder, f), out_dir)
        print("done:", f)
