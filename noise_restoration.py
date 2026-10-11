# __define-ocg__
# Noise models, noise estimation, mean filters, adaptive local noise reduction
# Works in color: every step runs on R, G and B separately
# Usage:
#   python noise_restoration.py images/
#   python noise_restoration.py images/ --mean 10 --var 0.8 --sp 0.05

import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

MEAN = 10.0     # noise mean (intensity levels)
VAR = 400.0     # noise variance (std = 20)
SP = 0.05       # salt-and-pepper density (fraction of pixels hit)
SALT = 0.5      # share of hits that are salt (white)
PATCH = 32      # flat-patch size for estimation
WIN = 3         # mean-filter window
ADAPT_WIN = 7   # adaptive-filter window
Q = 1.5         # contraharmonic order (> 0 removes pepper, < 0 removes salt)
CROP = 220      # zoom window for figures
CH = ["R", "G", "B"]
rng = np.random.default_rng(0)


# ---------- 1. Load color image, values 0-255 ----------
def load(path):
    img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return img.astype(np.float64)


# ---------- 2. Noise generators, each set by mean and variance ----------
# shape (H, W, 3): an independent value per pixel per channel
def gaussian(shape, mu, var):
    return rng.normal(mu, np.sqrt(var), shape)

def uniform(shape, mu, var):
    # uniform on [a, b]: mean (a+b)/2, variance (b-a)^2 / 12
    half = np.sqrt(3 * var)
    return rng.uniform(mu - half, mu + half, shape)

def rayleigh(shape, mu, var):
    # p(z) = 2/b (z-a) exp(-(z-a)^2 / b), z >= a
    # mean = a + sqrt(pi b / 4), variance = b (4 - pi) / 4
    b = 4 * var / (4 - np.pi)
    a = mu - np.sqrt(np.pi * b / 4)
    return a + np.sqrt(-b * np.log(1 - rng.random(shape)))

def gamma(shape, mu, var):
    # Erlang/gamma: mean = b/a, variance = b/a^2  (needs mean > 0)
    if mu <= 0:
        raise ValueError("Gamma noise needs a positive mean")
    a, b = mu / var, mu ** 2 / var
    return rng.gamma(b, 1 / a, shape)

def salt_pepper(img, density=SP, salt=SALT):
    # impulse noise: a fraction of samples forced to 0 (pepper) or 255 (salt), per channel
    out = img.copy()
    hit = rng.random(img.shape)
    out[hit < density * salt] = 255
    out[(hit >= density * salt) & (hit < density)] = 0
    return out

ADDITIVE = {"Gaussian": gaussian, "Uniform": uniform, "Rayleigh": rayleigh, "Gamma": gamma}


# ---------- 3. Add every noise type to one image ----------
def add_all_noise(img, mu=MEAN, var=VAR, sp=SP):
    noisy = {name: np.clip(img + fn(img.shape, mu, var), 0, 255) for name, fn in ADDITIVE.items()}
    noisy["Salt & pepper"] = salt_pepper(img, sp)
    return noisy


# ---------- 4. Estimate noise from a single image, per channel ----------
def box(x, k):
    # local mean over a k x k window (each channel separately)
    return cv2.blur(x, (k, k), borderType=cv2.BORDER_REFLECT)

def flattest_patch(img, size=PATCH):
    # window with the lowest local variance summed over R, G, B = least scene detail
    # every channel must be mid-range: near 0 or 255, clipping hides part of the noise
    mean = box(img, size)
    var = (box(img ** 2, size) - mean ** 2).sum(axis=-1)
    var[((mean < 60) | (mean > 195)).any(axis=-1)] = np.inf
    h, w = var.shape
    var[: size, :] = var[h - size:, :] = np.inf
    var[:, : size] = var[:, w - size:] = np.inf
    cy, cx = np.unravel_index(np.argmin(var), var.shape)
    return slice(cy - size // 2, cy + size // 2), slice(cx - size // 2, cx + size // 2)

def immerkaer_sigma(ch):
    # noise std from one whole channel: a mask that cancels smooth structure
    N = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    h, w = ch.shape
    s = np.abs(cv2.filter2D(ch, -1, N))[1:-1, 1:-1].sum()
    return np.sqrt(np.pi / 2) * s / (6 * (w - 2) * (h - 2))

def skewness(x):
    x = x.ravel()
    return float(np.mean((x - x.mean()) ** 3) / (x.std() ** 3 + 1e-12))

def estimate(noisy):
    ys, xs = flattest_patch(noisy)
    patch = noisy[ys, xs]                              # (32, 32, 3)
    return {
        "patch": (ys, xs),
        "patch_mean": patch.mean(axis=(0, 1)),         # one value per channel
        "patch_var": patch.var(axis=(0, 1)),           # variance estimate per channel
        "skew": np.array([skewness(patch[..., c]) for c in range(3)]),
        "immerkaer_var": np.array([immerkaer_sigma(noisy[..., c]) ** 2 for c in range(3)]),
        "impulse_density": impulse_density(noisy),
    }

def impulse_density(noisy):
    # impulse = at 0 or 255 AND far from its 3x3 median;
    # saturated clothing (red shirt, black top) sits at 0/255 too, but matches its neighbors
    med = cv2.medianBlur(to_u8(noisy), 3).astype(np.float64)
    extreme = (noisy == 0) | (noisy == 255)
    return np.mean(extreme & (np.abs(noisy - med) > 60), axis=(0, 1))


# ---------- 5. Mean filters (applied to each channel) ----------
def arithmetic(g, k=WIN):
    return box(g, k)

def weighted(g):
    W = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64) / 16
    return cv2.filter2D(g, -1, W, borderType=cv2.BORDER_REFLECT)

def geometric(g, k=WIN):
    # (product of pixels)^(1/mn) = exp(mean of logs)
    return np.exp(box(np.log(np.maximum(g, 1)), k))

def harmonic(g, k=WIN):
    # mn / sum(1/g)
    return 1 / box(1 / np.maximum(g, 1), k)

def contraharmonic(g, k=WIN, q=Q):
    # sum(g^(Q+1)) / sum(g^Q)
    g = np.maximum(g, 1)
    return box(g ** (q + 1), k) / box(g ** q, k)

MEAN_FILTERS = {"Arithmetic": arithmetic, "Weighted": weighted, "Geometric": geometric,
                "Harmonic": harmonic, f"Contraharmonic Q={Q}": contraharmonic}


# ---------- 6. Adaptive, local noise reduction filter (per channel) ----------
# f = g - (noise_var / local_var) * (g - local_mean), ratio capped at 1
#   noise_var = 0             -> returns g
#   local_var >> noise_var    -> close to g (edges kept)
#   local_var == noise_var    -> local mean
def adaptive(g, noise_var, k=ADAPT_WIN):
    m_L = box(g, k)
    var_L = np.maximum(box(g ** 2, k) - m_L ** 2, 1e-6)
    ratio = np.minimum(noise_var / var_L, 1.0)         # noise_var has one value per channel
    return g - ratio * (g - m_L), ratio


# ---------- 7. Figures ----------
def detail_window(img, size=CROP):
    # zoom on the most detailed region (usually hands and face)
    gray = img.mean(axis=-1)
    edges = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0)) + np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    score = cv2.boxFilter(edges, -1, (size, size))
    h, w = gray.shape
    score[: size // 2, :] = score[h - size // 2:, :] = 0
    score[:, : size // 2] = score[:, w - size // 2:] = 0
    cy, cx = np.unravel_index(score.argmax(), score.shape)
    return slice(cy - size // 2, cy + size // 2), slice(cx - size // 2, cx + size // 2)

def to_u8(x):
    return np.clip(x, 0, 255).astype(np.uint8)

def grid(rows, col_titles, path, crop):
    ys, xs = crop
    fig, axes = plt.subplots(len(rows), len(col_titles), figsize=(3 * len(col_titles), 3 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (row_name, ims) in enumerate(rows.items()):
        for c, im in enumerate(ims):
            axes[r, c].imshow(to_u8(im[ys, xs]))
            axes[r, c].set_title(f"{row_name}: {col_titles[c]}", fontsize=8)
            axes[r, c].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=80)
    plt.close(fig)


# ---------- 8. Process one image ----------
def process(path, out_dir, mu, var, sp):
    name = os.path.splitext(os.path.basename(path))[0]
    clean = load(path)
    crop = detail_window(clean)
    noisy = add_all_noise(clean, mu, var, sp)

    # noisy images
    grid({"Noise": [clean] + list(noisy.values())}, ["Clean"] + list(noisy),
         os.path.join(out_dir, f"{name}_noise.png"), crop)

    # estimation: flat-patch histogram per channel
    est = {k: estimate(v) for k, v in noisy.items()}
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.2))
    for ax, (k, e) in zip(axes, est.items()):
        ys, xs = e["patch"]
        for c, col in enumerate(["red", "green", "blue"]):
            ax.hist(noisy[k][ys, xs, c].ravel(), bins=50, color=col, alpha=0.45)
        ax.set_title(f"{k}\nvar R/G/B {' / '.join(f'{v:.0f}' for v in e['patch_var'])}", fontsize=9)
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_estimate.png"), dpi=80)
    plt.close(fig)

    # mean filters
    rows = {k: [v] + [f(v) for f in MEAN_FILTERS.values()] for k, v in noisy.items()}
    grid(rows, ["Noisy"] + list(MEAN_FILTERS), os.path.join(out_dir, f"{name}_mean_filters.png"), crop)

    # adaptive filter, using each channel's estimated noise variance
    rows = {}
    for k, v in noisy.items():
        out, ratio = adaptive(v, est[k]["patch_var"])
        rows[k] = [v, arithmetic(v, ADAPT_WIN), out, ratio * 255]
    grid(rows, ["Noisy", f"Arithmetic {ADAPT_WIN}x{ADAPT_WIN}", "Adaptive", "Ratio noise/local var"],
         os.path.join(out_dir, f"{name}_adaptive.png"), crop)
    return name, est


# ---------- 9. Run folder, print estimates per channel ----------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--mean", type=float, default=MEAN)
    p.add_argument("--var", type=float, default=VAR)
    p.add_argument("--sp", type=float, default=SP)
    args = p.parse_args()

    exts = (".png", ".jpg", ".jpeg", ".bmp")
    files = sorted(f for f in os.listdir(args.folder) if f.lower().endswith(exts))
    out_dir = "noise_results"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Set: mean {args.mean}, variance {args.var}, salt & pepper density {args.sp}\n")
    print(f"{'Image':8s} {'Noise':14s} {'Ch':3s} {'Patch var':>10s} {'Immerkaer var':>14s} {'Skew':>6s} {'Impulse %':>10s}")
    for f in files:
        name, est = process(os.path.join(args.folder, f), out_dir, args.mean, args.var, args.sp)
        for k, e in est.items():
            for c in range(3):
                print(f"{name:8s} {k:14s} {CH[c]:3s} {e['patch_var'][c]:10.1f} {e['immerkaer_var'][c]:14.1f} "
                      f"{e['skew'][c]:6.2f} {100 * e['impulse_density'][c]:10.2f}")
