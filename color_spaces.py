# __define-ocg__
# Color spaces: RGB, HSV, CIE-XYZ, Lab, Luv, YIQ, YUV
# Grayscale: classical, LIP and PLIP versions of 0.2989 R + 0.5870 G + 0.1140 B
# Usage: python color_spaces.py images/

import sys
import os
import colorsys
import numpy as np
import cv2
import matplotlib.pyplot as plt

EPS = 1e-10
WHITE_D65 = np.array([0.95047, 1.00000, 1.08883])   # reference white (Xn, Yn, Zn)
GRAY_W = (0.2989, 0.5870, 0.1140)                   # R, G, B weights for grayscale

# LIP / PLIP parameters (from the LIP homework)
M = 256.0
GAMMA = 1026.0


# ---------- 1. Load image as RGB in [0, 1] ----------
def load(path):
    img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return img.astype(np.float64) / 255.0


# ---------- 2. RGB: gamma-encoded (sRGB) <-> linear light ----------
# camera RGB is gamma-encoded; physics-based spaces (XYZ) need linear light
def srgb_to_linear(rgb):
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


# ---------- 3. HSV (artistic: hue, saturation, value) ----------
def rgb_to_hsv(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    v = rgb.max(axis=-1)                      # value = brightest channel
    c = v - rgb.min(axis=-1)                  # chroma = max - min
    s = np.where(v > 0, c / (v + EPS), 0)     # saturation = chroma relative to value
    # hue = angle on the color wheel, depends on which channel is max
    h = np.where(v == r, ((g - b) / (c + EPS)) % 6,
        np.where(v == g, (b - r) / (c + EPS) + 2,
                         (r - g) / (c + EPS) + 4)) / 6.0
    h = np.where(c == 0, 0, h)                # gray pixels have no hue
    return np.stack([h, s, v], axis=-1)


# ---------- 4. CIE-XYZ (standard, device-independent) ----------
# linear sRGB -> XYZ using the D65 matrix
RGB2XYZ = np.array([[0.4124, 0.3576, 0.1805],
                    [0.2126, 0.7152, 0.0722],
                    [0.0193, 0.1192, 0.9505]])

def rgb_to_xyz(rgb):
    return srgb_to_linear(rgb) @ RGB2XYZ.T


# ---------- 5. Perceptual: CIE Lab and CIE Luv ----------
# equal distances ~ equal perceived color differences
def _f(t):
    d = 6 / 29
    return np.where(t > d ** 3, np.cbrt(t), t / (3 * d ** 2) + 4 / 29)

def xyz_to_lab(xyz):
    fx, fy, fz = [_f(xyz[..., i] / WHITE_D65[i]) for i in range(3)]
    L = 116 * fy - 16              # lightness 0-100
    a = 500 * (fx - fy)            # green (-) to red (+)
    b = 200 * (fy - fz)            # blue (-) to yellow (+)
    return np.stack([L, a, b], axis=-1)

def xyz_to_luv(xyz):
    X, Y, Z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    Xn, Yn, Zn = WHITE_D65
    L = 116 * _f(Y / Yn) - 16
    den = X + 15 * Y + 3 * Z + EPS
    un = 4 * Xn / (Xn + 15 * Yn + 3 * Zn)
    vn = 9 * Yn / (Xn + 15 * Yn + 3 * Zn)
    u = 13 * L * (4 * X / den - un)   # chromaticity offsets from white
    v = 13 * L * (9 * Y / den - vn)
    return np.stack([L, u, v], axis=-1)


# ---------- 6. Opponent: YIQ and YUV ----------
# Y = brightness, other two = color-difference (opponent) channels
def rgb_to_yiq(rgb):
    # same coefficients as Python's colorsys
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.30 * r + 0.59 * g + 0.11 * b
    i = 0.74 * (r - y) - 0.27 * (b - y)       # orange-blue axis
    q = 0.48 * (r - y) + 0.41 * (b - y)       # purple-green axis
    return np.stack([y, i, q], axis=-1)

def rgb_to_yuv(rgb):
    # ITU-R BT.601
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = 0.492 * (b - y)                       # blue difference
    v = 0.877 * (r - y)                       # red difference
    return np.stack([y, u, v], axis=-1)


# ---------- 7. Grayscale: classical, LIP and PLIP ----------
def gray_classical(rgb):
    # gray = 0.2989 R + 0.5870 G + 0.1140 B
    return GRAY_W[0] * rgb[..., 0] + GRAY_W[1] * rgb[..., 1] + GRAY_W[2] * rgb[..., 2]

def _log_gray(rgb, mu):
    # gray = (w_R ⊗ R) ⊕ (w_G ⊗ G) ⊕ (w_B ⊗ B), using gray tones t = M - I
    t = M - rgb * 255.0
    out = np.zeros(rgb.shape[:2])
    for w, ch in zip(GRAY_W, range(3)):
        scaled = mu - mu * (1 - t[..., ch] / mu) ** w     # w ⊗ channel
        out = out + scaled - out * scaled / mu            # ⊕ running total
    return np.clip((M - out) / 255.0, 0, 1)

def gray_lip(rgb):
    return _log_gray(rgb, M)

def gray_plip(rgb):
    return _log_gray(rgb, GAMMA)


# ---------- 8. Verify against Python's colorsys ----------
def verify():
    px = np.random.default_rng(0).random((500, 3))
    ref_hsv = np.array([colorsys.rgb_to_hsv(*p) for p in px])
    ref_yiq = np.array([colorsys.rgb_to_yiq(*p) for p in px])
    print("Max error vs colorsys:")
    print(f"  HSV  {np.abs(rgb_to_hsv(px) - ref_hsv).max():.2e}")
    print(f"  YIQ  {np.abs(rgb_to_yiq(px) - ref_yiq).max():.2e}")


# ---------- 9. Process one image, run the folder ----------
SPACES = {
    "RGB": (lambda x: x,                     ["R", "G", "B"]),
    "HSV": (rgb_to_hsv,                      ["H", "S", "V"]),
    "XYZ": (rgb_to_xyz,                      ["X", "Y", "Z"]),
    "Lab": (lambda x: xyz_to_lab(rgb_to_xyz(x)), ["L", "a", "b"]),
    "Luv": (lambda x: xyz_to_luv(rgb_to_xyz(x)), ["L", "u", "v"]),
    "YIQ": (rgb_to_yiq,                      ["Y", "I", "Q"]),
    "YUV": (rgb_to_yuv,                      ["Y", "U", "V"]),
}

def show(ch):
    # stretch one channel to 0-1 for display (channels have different ranges)
    lo, hi = ch.min(), ch.max()
    return (ch - lo) / (hi - lo + EPS)

def combined(conv):
    # all three channels together, displayed as R, G, B (false color)
    return np.stack([show(conv[..., c]) for c in range(3)], axis=-1)

def process(path, out_dir):
    rgb = load(path)
    name = os.path.splitext(os.path.basename(path))[0]

    # figure 1: every color space, one row each:
    # original, combined (3 channels together), then each channel
    fig, axes = plt.subplots(len(SPACES), 5, figsize=(16, 3.2 * len(SPACES)))
    for r, (space, (fn, labels)) in enumerate(SPACES.items()):
        conv = fn(rgb)
        axes[r, 0].imshow(rgb)
        axes[r, 0].set_title(f"{space}: original")
        axes[r, 1].imshow(combined(conv))
        axes[r, 1].set_title(f"{space}: combined")
        for c in range(3):
            axes[r, c + 2].imshow(show(conv[..., c]), cmap="gray")
            axes[r, c + 2].set_title(f"{space}: {labels[c]}")
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_spaces.png"), dpi=80)
    plt.close(fig)

    # figure 2: grayscale comparison
    grays = {"Original": rgb, "Classical gray": gray_classical(rgb),
             "LIP gray": gray_lip(rgb), "PLIP gray": gray_plip(rgb)}
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, (title, im) in zip(axes, grays.items()):
        ax.imshow(im, cmap=None if im.ndim == 3 else "gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_gray.png"), dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    verify()
    folder = sys.argv[1]
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(exts))
    print(f"\nFound {len(files)} images")

    out_dir = "color_results"
    os.makedirs(out_dir, exist_ok=True)
    for f in files:
        process(os.path.join(folder, f), out_dir)
        print("done:", f)
