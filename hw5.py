# __define-ocg__
# Lecture 5: noise models (Gaussian, uniform, Poisson, salt & pepper), noise estimation, order-statistic filters,
# adaptive local noise reduction (median / alpha-trimmed mean), order-statistic sharpening,
# bonus: Non-Local Means denoising with OpenCV (experiments from the OpenCV tutorial)
# Gray-level images
# Usage:
#   python hw5.py images/
#   python hw5.py images/ --mean 10 --var 0.8 --sp 0.05

import os
import json
import time
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import optimize, special

MEAN = 10.0      # noise mean (intensity levels)
VAR = 400.0      # noise variance (std = 20)
SP = 0.05        # salt-and-pepper density
SALT = 0.5       # share of hits that are salt
REGION = 128     # uniform-region size for estimation
WIN = 3          # order-statistic filter window
ALPHA_D = 4      # alpha-trimmed mean: drop d/2 lowest and d/2 highest
WMED = np.array([[1, 2, 1], [2, 3, 2], [1, 2, 1]])   # weighted median: copies of each pixel
ADAPT_WIN = 5    # adaptive filter window
BLUR = 1.2       # Gaussian blur (sigma) before sharpening
SHARP_NOISE = 5  # noise std added to the blurred image in the sharpening test
CROP = 220       # zoom window for figures
NLM_PATCH = 7    # OpenCV NLM templateWindowSize (tutorial default)
NLM_SEARCH = 21  # OpenCV NLM searchWindowSize (tutorial default)
NLM_H_TUTORIAL = 10                      # h used in the OpenCV tutorial
H_SWEEP = [5, 10, 15, 20, 25, 30, 40]    # filter strengths tried
TEMPLATES = [3, 5, 7, 11]                # patch sizes tried
SEARCHES = [7, 11, 21, 35]               # search window sizes tried
FRAMES = 5                               # noisy frames for fastNlMeansDenoisingMulti
rng = np.random.default_rng(0)


# ---------- 1. Load as gray, values 0-255 ----------
def load(path):
    return cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY).astype(np.float64)

def to_u8(x):
    return np.clip(x, 0, 255).astype(np.uint8)


# ---------- 2. Noise generators, each set by mean and variance ----------
def gaussian(shape, mu, var):
    return rng.normal(mu, np.sqrt(var), shape)

def uniform(shape, mu, var):
    # uniform on [a, b]: mean (a+b)/2, variance (b-a)^2 / 12
    half = np.sqrt(3 * var)
    return rng.uniform(mu - half, mu + half, shape)

def poisson(img, mu, var):
    # photon (shot) noise: signal-dependent, variance proportional to intensity
    # lam = photons per gray level, chosen so the variance averages `var` over the image; then shifted by `mu`
    lam = img.mean() / var
    return rng.poisson(img * lam) / lam - img + mu

def salt_pepper(img, density=SP, salt=SALT):
    # a fraction of pixels forced to 255 (salt) or 0 (pepper)
    out = img.copy()
    hit = rng.random(img.shape)
    out[hit < density * salt] = 255
    out[(hit >= density * salt) & (hit < density)] = 0
    return out

ADDITIVE = {"Gaussian": gaussian, "Uniform": uniform}

def add_all_noise(img, mu=MEAN, var=VAR, sp=SP):
    # rounded and clipped, like a real 8-bit image
    noisy = {name: np.round(np.clip(img + fn(img.shape, mu, var), 0, 255)) for name, fn in ADDITIVE.items()}
    noisy["Poisson"] = np.round(np.clip(img + poisson(img, mu, var), 0, 255))
    noisy["Salt & pepper"] = salt_pepper(img, sp)
    return noisy


# ---------- 3. Estimate noise from a single image (lecture steps 1-6) ----------
def box(x, k):
    return cv2.blur(x, (k, k), borderType=cv2.BORDER_REFLECT)

# Step 1: segment (k-means on intensity; median + blur first so noise doesn't split segments)
def segment(g, k=3):
    small = cv2.resize(g, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(cv2.medianBlur(to_u8(small), 5).astype(np.float32), (0, 0), 2)
    _, lab, _ = cv2.kmeans(small.reshape(-1, 1), k, None,
                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5), 5, cv2.KMEANS_PP_CENTERS)
    lab = lab.reshape(small.shape)
    labels = cv2.resize(lab.astype(np.uint8), g.shape[::-1], interpolation=cv2.INTER_NEAREST)
    largest = np.bincount(lab.ravel()).argmax()
    return labels, (labels == largest).astype(np.uint8)

# Step 2: large uniform region = lowest-variance window fully inside the largest segment
def uniform_region(g, mask):
    for size in (REGION, 96, 64, 32):
        inside = cv2.erode(mask, np.ones((size + 1, size + 1), np.uint8)) > 0
        inside[: size // 2 + 1] = inside[-size // 2 - 1:] = False
        inside[:, : size // 2 + 1] = inside[:, -size // 2 - 1:] = False
        if inside.any():
            break
    var = box(g ** 2, size) - box(g, size) ** 2
    var[~inside] = np.inf
    cy, cx = np.unravel_index(np.argmin(var), var.shape)
    return slice(cy - size // 2, cy + size // 2), slice(cx - size // 2, cx + size // 2)

# Step 3: histogram h(x_i)
def histogram(values):
    return np.bincount(np.clip(np.round(values), 0, 255).astype(int).ravel(), minlength=256)

# Step 4: mean and variance of the histogram
def hist_moments(h):
    x = np.arange(256)
    p = h / h.sum()
    mu = (x * p).sum()
    var = ((x - mu) ** 2 * p).sum()
    skew = ((x - mu) ** 3 * p).sum() / var ** 1.5
    return mu, var, skew

# Step 5: distribution from the shape (maximum likelihood; bins 0 and 255 hold clipped tails)
EDGES = np.arange(257) - 0.5
EDGES[0], EDGES[-1] = -np.inf, np.inf
NPAR = {"Gaussian": 2, "Uniform": 2, "Rayleigh": 2, "Gamma": 3}

def cdf(name, th, x):
    if name == "Gaussian":
        return special.ndtr((x - th[0]) / abs(th[1]))
    if name == "Uniform":
        return np.clip((x - th[0]) / abs(th[1]), 0, 1)
    z = np.maximum(x - th[0], 0)
    if name == "Rayleigh":
        return 1 - np.exp(-(z / abs(th[1])) ** 2 / 2)
    return special.gammainc(abs(th[2]), z / abs(th[1]))

def neg_log_lik(th, name, h):
    q = np.clip(np.diff(cdf(name, th, EDGES)), 1e-12, None)
    return -(h * np.log(q)).sum()

def start(name, mu, var, skew):
    sd = np.sqrt(var)
    if name == "Gaussian":
        return [mu, sd]
    if name == "Uniform":
        return [mu - np.sqrt(3) * sd, 2 * np.sqrt(3) * sd]
    if name == "Rayleigh":
        s = sd / np.sqrt((4 - np.pi) / 2)
        return [mu - s * np.sqrt(np.pi / 2), s]
    k = 4 / max(skew, 0.3) ** 2
    return [mu - k * sd / np.sqrt(k), sd / np.sqrt(k), k]

def fit_variance(name, th):
    if name == "Gaussian":
        return th[1] ** 2
    if name == "Uniform":
        return th[1] ** 2 / 12
    if name == "Rayleigh":
        return th[1] ** 2 * (4 - np.pi) / 2
    return abs(th[2]) * th[1] ** 2

def classify(h):
    N = h.sum()
    if h[0] / N > 0.002 and h[255] / N > 0.002:
        return "Salt & pepper", None           # spikes at both ends of a mid-range region
    fits = {}
    mu, var, skew = hist_moments(h)
    for name in NPAR:
        r = optimize.minimize(neg_log_lik, start(name, mu, var, skew), args=(name, h),
                              method="Nelder-Mead", options=dict(maxiter=2000, xatol=1e-3, fatol=1e-3))
        fits[name] = (2 * r.fun + NPAR[name] * np.log(N), r.x)     # BIC
    best = min(fits, key=lambda n: fits[n][0])
    if best == "Gamma":
        k = abs(fits["Gamma"][1][2])
        two = min((n for n in fits if NPAR[n] == 2), key=lambda n: fits[n][0])
        if abs(2 / np.sqrt(k) - 0.63) < 0.3:
            best = "Rayleigh"
        elif fits[two][0] - fits["Gamma"][0] < 10:
            best = two
    return best, fits[best][1]

# Step 6: the distribution's parameters
def parameters(name, h, th):
    mu, var, skew = hist_moments(h)
    N = h.sum()
    clipped = (h[0] + h[255]) / N
    if name == "Salt & pepper":
        return {"Pa (pepper)": h[0] / N, "Pb (salt)": h[255] / N}, var
    v = var if clipped <= 0.005 or name == "Gamma" else fit_variance(name, th)
    if name == "Gaussian":
        return {"mean": mu, "var": v}, v
    if name == "Uniform":
        half = np.sqrt(3 * v)
        return {"a": mu - half, "b": mu + half}, v
    if name == "Rayleigh":
        b = 4 * v / (4 - np.pi)
        return {"a": mu - np.sqrt(np.pi * b / 4), "b": b}, v
    k = 4 / skew ** 2
    return {"a": np.sqrt(k / v), "b": k}, v

# Extra check: does the noise grow with brightness? (signal-dependent noise, e.g. Poisson)
# one flat window per segment gives (mean, variance) points; Poisson: variance rises along a line
def flat_points(g, k=6, size=32):
    labels, _ = segment(g, k)
    var_map = box(g ** 2, size) - box(g, size) ** 2
    pts = []
    for L in range(k):
        inside = cv2.erode((labels == L).astype(np.uint8), np.ones((size + 1, size + 1), np.uint8)) > 0
        inside[: size // 2 + 1] = inside[-size // 2 - 1:] = False
        inside[:, : size // 2 + 1] = inside[:, -size // 2 - 1:] = False
        if not inside.any():
            continue
        v = np.where(inside, var_map, np.inf)
        cy, cx = np.unravel_index(np.argmin(v), v.shape)
        w = g[cy - size // 2:cy + size // 2, cx - size // 2:cx + size // 2]
        if ((w == 0) | (w == 255)).mean() < 0.01:                     # skip clipped windows
            pts.append((w.mean(), w.var()))
    return np.array(pts)

def signal_dependent(pts):
    # line var = a * mean + b; signal-dependent if the variance changes by more than half its size across the means
    if len(pts) < 2:
        return False, 0.0, float(pts[:, 1].mean()) if len(pts) else 0.0
    a, b = np.polyfit(pts[:, 0], pts[:, 1], 1)
    rise = a * (pts[:, 0].max() - pts[:, 0].min())
    return bool(a > 0 and rise > 0.5 * np.median(pts[:, 1])), float(a), float(b)

def estimate(noisy):
    labels, mask = segment(noisy)                    # step 1
    ys, xs = uniform_region(noisy, mask)             # step 2
    h = histogram(noisy[ys, xs])                     # step 3
    name, th = classify(h)                           # steps 4-5
    par, v = parameters(name, h, th)                 # step 6
    pts = flat_points(noisy)
    dep, a, b = signal_dependent(pts)
    if dep and name != "Salt & pepper":
        # Poisson: variance = a * intensity (+ b); report a and the variance at the image's mean intensity
        name = "Poisson"
        par = {"var per level a": a, "var at mean level": a * noisy.mean() + b}
    return {"labels": labels, "region": (ys, xs), "hist": h, "type": name, "params": par, "var": float(v),
            "points": pts.tolist(), "slope": a}


# ---------- 4. Order-statistic filters ----------
def neighbours(g, k=WIN, weights=None):
    # sorted values of each pixel's k x k neighbourhood, shape (n, H, W); weights = copies per position
    r = k // 2
    p = np.pad(g, r, mode="reflect")
    H, W = g.shape
    vals = []
    for dy in range(k):
        for dx in range(k):
            copies = 1 if weights is None else int(weights[dy, dx])
            vals += [p[dy:dy + H, dx:dx + W]] * copies
    return np.sort(np.stack(vals), axis=0)

def median(g, k=WIN):
    s = neighbours(g, k)
    return s[len(s) // 2]

def max_filter(g, k=WIN):
    return neighbours(g, k)[-1]          # removes pepper, grows bright regions

def min_filter(g, k=WIN):
    return neighbours(g, k)[0]           # removes salt, grows dark regions

def weighted_median(g, w=WMED):
    s = neighbours(g, w.shape[0], w)     # center counted 3 times, edges 2, corners 1
    return s[len(s) // 2]

def midpoint(g, k=WIN):
    s = neighbours(g, k)
    return (s[0] + s[-1]) / 2            # best for uniform noise; ruined by impulses

def alpha_trimmed(g, k=WIN, d=ALPHA_D):
    s = neighbours(g, k)
    return s[d // 2: len(s) - d // 2].mean(axis=0)   # mean of the middle mn - d values

ORDER_FILTERS = {"Median": median, "Max": max_filter, "Min": min_filter, "Weighted median": weighted_median,
                 "Midpoint": midpoint, "Alpha-trimmed": alpha_trimmed}


# ---------- 5. Adaptive local noise reduction ----------
# f = g - (var_n / var_L) (g - m_L),  m_L = local median or alpha-trimmed mean
#   var_n = 0        -> f = g
#   var_L >> var_n   -> f close to g (edges kept)
#   var_L == var_n   -> f = m_L
def adaptive(g, var_n, centre="median", k=ADAPT_WIN):
    mu = box(g, k)
    var_l = np.maximum(box(g ** 2, k) - mu ** 2, 1e-6)                 # local variance
    m_l = median(g, k) if centre == "median" else alpha_trimmed(g, k, d=k * k // 2)
    ratio = np.clip(var_n / var_l, 0, 1)                                # never past m_L
    return g - ratio * (g - m_l)


# ---------- 6. Sharpening with order-statistic filters ----------
# unsharp masking: s = g + k (g - OS(g)); the detail g - OS(g) is added back
def unsharp(g, base, k=1.0):
    return g + k * (g - base(g))

def toggle_contrast(g, k=WIN):
    # each pixel jumps to the local max or min, whichever is closer: edges become steps
    s = neighbours(g, k)
    return np.where(s[-1] - g < g - s[0], s[-1], s[0])

SHARPENERS = {"Median": lambda g: unsharp(g, median), "Weighted median": lambda g: unsharp(g, weighted_median),
              "Midpoint": lambda g: unsharp(g, midpoint), "Alpha-trimmed": lambda g: unsharp(g, alpha_trimmed),
              "Max/min (toggle)": toggle_contrast}


# ---------- 6b. Bonus: Non-Local Means (OpenCV) ----------
# each pixel = weighted average of pixels whose surrounding patch looks similar, within a search window
# h: filter strength (larger = more smoothing); templateWindowSize: patch; searchWindowSize: search area
def nlm(g, h, template=NLM_PATCH, search=NLM_SEARCH):
    return cv2.fastNlMeansDenoising(to_u8(g), None, h, template, search).astype(np.float64)

def timed(fn, *a):
    t = time.time()
    out = fn(*a)
    return out, time.time() - t


# ---------- 7. Errors against the clean image ----------
def gradient(x):
    return np.hypot(cv2.Sobel(x, cv2.CV_64F, 1, 0), cv2.Sobel(x, cv2.CV_64F, 0, 1))

def edge_mask(clean):
    g = gradient(clean)
    return g > np.percentile(g, 90)          # strongest 10% of edges: fingers, face, hair

def errors_of(out, clean, edges):
    # MSE, MSE without the constant offset, MSE on edges (no offset), edge sharpness vs clean
    out = np.clip(out, 0, 255)
    err = out - clean
    e0 = err - err.mean()
    sharp = gradient(out)[edges].mean() / gradient(clean)[edges].mean()
    return float(np.mean(err ** 2)), float(np.mean(e0 ** 2)), float(np.mean(e0[edges] ** 2)), float(sharp)


# ---------- 8. Figures ----------
def detail_window(img, size=CROP):
    score = cv2.boxFilter(gradient(img), -1, (size, size))
    h, w = img.shape
    score[: size // 2, :] = score[h - size // 2:, :] = 0
    score[:, : size // 2] = score[:, w - size // 2:] = 0
    cy, cx = np.unravel_index(score.argmax(), score.shape)
    return slice(cy - size // 2, cy + size // 2), slice(cx - size // 2, cx + size // 2)

def grid(rows, col_titles, path, crop, errs=None):
    ys, xs = crop
    fig, axes = plt.subplots(len(rows), len(col_titles), figsize=(3 * len(col_titles), 3.3 * len(rows) + 0.4))
    axes = np.atleast_2d(axes)
    for r, (row_name, ims) in enumerate(rows.items()):
        for c, im in enumerate(ims):
            m = errs[row_name][c] if errs else None
            axes[r, c].imshow(to_u8(im[ys, xs]), cmap="gray", vmin=0, vmax=255)
            cap = "" if m is None else f"\nMSE = {m:.2f}, RMSE = {np.sqrt(m):.2f}"
            axes[r, c].set_title(f"{row_name}: {col_titles[c]}{cap}", fontsize=8)
            axes[r, c].axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=80)
    plt.close(fig)

def noise_figure(clean, noisy, errs, path, var):
    # image (MSE, RMSE in title) over its histogram with mean and mean +- std, and a grayscale bar
    ims = {"Original": clean, **noisy}
    hists = {k: np.bincount(to_u8(v).ravel(), minlength=256) for k, v in ims.items()}
    top = 1.1 * max(h[1:255].max() for k, h in hists.items() if k != "Original")
    fig, axes = plt.subplots(2, len(ims), figsize=(4 * len(ims), 7.4), gridspec_kw={"height_ratios": [1, 0.75]})
    for c, (k, v) in enumerate(ims.items()):
        x = to_u8(v).astype(float)
        mu, sd = x.mean(), x.std()
        title = k if k == "Original" else f"{k} (var {var:g})\nMSE:{errs[k]:.2f}, RMSE:{np.sqrt(errs[k]):.2f}"
        if k == "Salt & pepper":
            title = f"Salt & pepper ({SP:g})\nMSE:{errs[k]:.2f}, RMSE:{np.sqrt(errs[k]):.2f}"
        axes[0, c].imshow(x, cmap="gray", vmin=0, vmax=255)
        axes[0, c].set_title(title, fontsize=14)
        axes[0, c].axis("off")
        ax = axes[1, c]
        ax.bar(np.arange(256), hists[k], width=1, color="0.6")
        ax.axvline(mu, color="red", ls="--", label=f"μ={mu:.2f}")
        ax.axvline(mu - sd, color="green", ls=":", label=f"σ={sd:.2f}")
        ax.axvline(mu + sd, color="green", ls=":")
        ax.set_xlim(0, 256); ax.set_ylim(0, top)
        ax.set_ylabel("Frequency", fontsize=12); ax.set_xticks([]); ax.tick_params(labelsize=11)
        ax.legend(fontsize=12, loc="upper right")
        bar = ax.inset_axes([0, -0.2, 1, 0.1])
        bar.imshow(np.linspace(0, 1, 256)[None], cmap="gray", aspect="auto", extent=(0, 255, 0, 1))
        bar.set_yticks([]); bar.set_xticks([0, 255]); bar.tick_params(labelsize=11)
        bar.set_xlabel("Grayscale", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=80)
    plt.close(fig)


# ---------- 9. Process one image ----------
def process(path, out_dir, mu, var, sp):
    name = os.path.splitext(os.path.basename(path))[0]
    clean = load(path)
    crop = detail_window(clean)
    edges = edge_mask(clean)
    noisy = add_all_noise(clean, mu, var, sp)
    errors = {k: {"Noisy": errors_of(v, clean, edges)} for k, v in noisy.items()}

    # part 1: noisy images with histograms
    noise_figure(clean, noisy, {k: errors[k]["Noisy"][0] for k in noisy}, os.path.join(out_dir, f"{name}_noise.png"), var)

    # part 2: estimation
    est = {k: estimate(v) for k, v in noisy.items()}

    # part 3: order-statistic filters
    rows = {}
    for k, v in noisy.items():
        outs = [f(v) for f in ORDER_FILTERS.values()]
        rows[k] = [v] + outs
        errors[k].update({n: errors_of(o, clean, edges) for n, o in zip(ORDER_FILTERS, outs)})
    cols = ["Noisy"] + list(ORDER_FILTERS)
    grid(rows, cols, os.path.join(out_dir, f"{name}_order_filters.png"), crop,
         {k: [errors[k][n][0] for n in cols] for k in rows})

    # max removes pepper, min removes salt: one kind of impulse at a time
    pepper = salt_pepper(clean, sp, salt=0)
    salt = salt_pepper(clean, sp, salt=1)
    errors["Pepper only"] = {"Noisy": errors_of(pepper, clean, edges), "Max": errors_of(max_filter(pepper), clean, edges),
                             "Min": errors_of(min_filter(pepper), clean, edges)}
    errors["Salt only"] = {"Noisy": errors_of(salt, clean, edges), "Max": errors_of(max_filter(salt), clean, edges),
                           "Min": errors_of(min_filter(salt), clean, edges)}
    grid({"Pepper only": [pepper, max_filter(pepper), min_filter(pepper)],
          "Salt only": [salt, max_filter(salt), min_filter(salt)]}, ["Noisy", "Max", "Min"],
         os.path.join(out_dir, f"{name}_max_min.png"), crop,
         {r: [errors[r][n][0] for n in ["Noisy", "Max", "Min"]] for r in ["Pepper only", "Salt only"]})

    # part 4: adaptive filter with the estimated noise variance
    rows = {}
    for k, v in noisy.items():
        outs = {f"Median {ADAPT_WIN}x{ADAPT_WIN}": median(v, ADAPT_WIN),
                f"Adaptive (median)": adaptive(v, est[k]["var"], "median"),
                f"Adaptive (alpha-trimmed)": adaptive(v, est[k]["var"], "alpha")}
        rows[k] = [v] + list(outs.values())
        errors[k].update({n: errors_of(o, clean, edges) for n, o in outs.items()})
    cols = ["Noisy"] + list(outs)
    grid(rows, cols, os.path.join(out_dir, f"{name}_adaptive.png"), crop,
         {k: [errors[k][n][0] for n in cols] for k in rows})

    # part 5: sharpening a blurred frame, then a blurred and noisy one
    blurred = cv2.GaussianBlur(clean, (0, 0), BLUR)
    blur_noisy = np.round(np.clip(blurred + rng.normal(0, SHARP_NOISE, clean.shape), 0, 255))
    sharp_in = {"Blurred": blurred, "Blurred + noise": blur_noisy}
    rows = {}
    for k, v in sharp_in.items():
        outs = {n: f(v) for n, f in SHARPENERS.items()}
        if k == "Blurred + noise":
            outs["Median, then midpoint"] = unsharp(median(v), midpoint)       # denoise first
        rows[k] = [v] + list(outs.values())
        errors[k] = {"Input": errors_of(v, clean, edges), **{n: errors_of(o, clean, edges) for n, o in outs.items()}}
    for k in rows:
        cols = ["Input"] + list(errors[k])[1:]
        grid({k: rows[k]}, cols, os.path.join(out_dir, f"{name}_sharpen_{k.split()[0].lower()}{'_noise' if '+' in k else ''}.png"),
             crop, {k: [errors[k][n][0] for n in cols]})
    # part 6: what each filter does to the clean frame (no noise), and where the median leaves its error
    errors["Clean frame"] = {n: errors_of(f(clean), clean, edges) for n, f in ORDER_FILTERS.items()}
    errors["Clean frame"]["Adaptive (alpha-trimmed)"] = errors_of(adaptive(clean, est["Gaussian"]["var"], "alpha"), clean, edges)
    near = cv2.dilate(edges.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    errors["S&P error near edges"] = {}
    for n in ("Median", "Weighted median", "Alpha-trimmed"):
        e = (np.clip(ORDER_FILTERS[n](noisy["Salt & pepper"]), 0, 255) - clean) ** 2
        errors["S&P error near edges"][n] = (float(e[near].sum() / e.sum()), float(near.mean()))
    sharpening_diagrams(clean, noisy["Gaussian"], est["Gaussian"]["region"], name, out_dir)

    # bonus: OpenCV Non-Local Means
    bonus_rng = np.random.default_rng(100 + len(name))       # separate draws, so the results above are unchanged
    # 1) tutorial's h = 10 vs h = estimated noise std, against the best filters above
    rows = {}
    for k, v in noisy.items():
        sd = np.sqrt(est[k]["var"])
        outs = {f"NLM h={NLM_H_TUTORIAL} (tutorial)": nlm(v, NLM_H_TUTORIAL), "NLM h=estimated std": nlm(v, sd)}
        rows[k] = [v, alpha_trimmed(v), adaptive(v, est[k]["var"], "alpha")] + list(outs.values())
        errors[k].update({n: errors_of(o, clean, edges) for n, o in outs.items()})
    cols = ["Noisy", "Alpha-trimmed", "Adaptive (alpha-trimmed)"] + list(outs)
    grid(rows, cols, os.path.join(out_dir, f"{name}_nlm.png"), crop, {k: [errors[k][n][0] for n in cols] for k in rows})

    # 2) parameters, Gaussian noise: strength h, patch size, search window (with h = estimated std)
    g, sd = noisy["Gaussian"], np.sqrt(est["Gaussian"]["var"])
    sweeps = {"NLM h": {f"h={h}": (h, NLM_PATCH, NLM_SEARCH) for h in H_SWEEP},
              "NLM template": {f"template={t}": (sd, t, NLM_SEARCH) for t in TEMPLATES},
              "NLM search": {f"search={w}": (sd, NLM_PATCH, w) for w in SEARCHES}}
    for sweep, runs in sweeps.items():
        outs = {}
        errors[sweep] = {}
        for lab, par in runs.items():
            outs[lab], sec = timed(nlm, g, *par)
            errors[sweep][lab] = errors_of(outs[lab], clean, edges) + (sec,)
        grid({sweep: [g] + list(outs.values())}, ["Noisy"] + list(outs), os.path.join(out_dir, f"{name}_{sweep.replace(' ', '_').lower()}.png"),
             crop, {sweep: [errors["Gaussian"]["Noisy"][0]] + [errors[sweep][l][0] for l in outs]})

    # 3) fastNlMeansDenoisingMulti: several noisy frames of a still scene, denoise the middle one
    seq = [to_u8(np.round(np.clip(clean + bonus_rng.normal(mu, np.sqrt(var), clean.shape), 0, 255))) for _ in range(FRAMES)]
    mid = FRAMES // 2
    outs = {"Single frame": nlm(seq[mid], sd)}
    for tw in (3, 5):
        outs[f"Multi, {tw} frames"] = cv2.fastNlMeansDenoisingMulti(seq, mid, tw, None, sd, NLM_PATCH, NLM_SEARCH).astype(np.float64)
    errors["NLM multi"] = {"Noisy": errors_of(seq[mid].astype(float), clean, edges),
                           **{n: errors_of(o, clean, edges) for n, o in outs.items()}}
    grid({"NLM multi": [seq[mid]] + list(outs.values())}, ["Noisy"] + list(outs), os.path.join(out_dir, f"{name}_nlm_multi.png"),
         crop, {"NLM multi": [errors["NLM multi"][n][0] for n in ["Noisy"] + list(outs)]})

    # 4) fastNlMeansDenoisingColored on the color frame (Gaussian noise): h for luminance, hColor for color
    rgb = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float64)
    noisy_rgb = to_u8(np.round(np.clip(rgb + bonus_rng.normal(mu, np.sqrt(var), rgb.shape), 0, 255)))
    color_mse = lambda o: float(np.mean((o.astype(float) - rgb) ** 2))
    color_mse0 = lambda o: float(np.mean(((o.astype(float) - rgb) - (o.astype(float) - rgb).mean(axis=(0, 1))) ** 2))
    outs = {"Each channel as gray": np.stack([nlm(noisy_rgb[..., c], sd) for c in range(3)], axis=-1)}
    for hc in (10, 20, 40):
        outs[f"Colored, hColor={hc}"] = cv2.fastNlMeansDenoisingColored(noisy_rgb, None, sd, hc, NLM_PATCH, NLM_SEARCH).astype(np.float64)
    errors["NLM color"] = {"Noisy": (color_mse(noisy_rgb), color_mse0(noisy_rgb)),
                           **{n: (color_mse(o), color_mse0(o)) for n, o in outs.items()}}
    ys, xs = crop
    fig, axes = plt.subplots(1, 1 + len(outs), figsize=(3 * (1 + len(outs)), 3.7))
    for ax, (lab, im) in zip(axes, {"Noisy": noisy_rgb, **outs}.items()):
        e = errors["NLM color"][lab][0]
        ax.imshow(to_u8(im[ys, xs]))
        ax.set_title(f"{lab}\nMSE = {e:.2f}, RMSE = {np.sqrt(e):.2f}", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_nlm_color.png"), dpi=80)
    plt.close(fig)
    return name, est, errors


# ---------- 9b. Part 5 diagrams on a real frame: an order-statistic window, and sharpening an edge ----------
def sharpening_diagrams(clean, noisy_gauss, region, name, out_dir):
    # 1) a 3 x 3 window in the flat region of the Gaussian-noisy frame, plus one pepper impulse, sorted
    ys, xs = region
    y, x = (ys.start + ys.stop) // 2, (xs.start + xs.stop) // 2
    w = noisy_gauss[y - 1:y + 2, x - 1:x + 2].copy()
    ref = w.mean()                                                 # window mean before the impulse
    w[1, 2] = 0                                                    # one pepper impulse
    win = np.sort(w.ravel())
    d = ALPHA_D // 2
    kept = win[d:len(win) - d]
    top = 290
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(np.arange(1, 10), win, color=["0.75"] * d + ["0.25"] * (9 - 2 * d) + ["0.75"] * d, width=0.6)
    for k, v in enumerate(win, 1):
        ax.text(k, v + 4, f"{v:.0f}", ha="center", fontsize=8)
    for xx in (d + 0.6, 9 - d + 0.4):
        ax.plot([xx, xx], [0, 262], color="red", ls="--", lw=1.5)
    ax.plot([d + 0.6, 9 - d + 0.4], [262, 262], color="red", lw=2)
    ax.text(1.5, 270, "trimmed (low)", ha="center", fontsize=9)
    ax.text(5, 270, "kept: averaged", ha="center", fontsize=9, color="red")
    ax.text(8.5, 270, "trimmed (high)", ha="center", fontsize=9)
    ax.axhline(ref, color="0.6", lw=3, alpha=0.5, label=f"mean without the impulse = {ref:.0f}")
    for val, lab, col, ls in [(win.mean(), "mean", "0.4", "-"), (kept.mean(), "alpha-trimmed mean", "tab:blue", "-"),
                              (win[4], "median", "tab:green", ":"), ((win[0] + win[-1]) / 2, "midpoint", "tab:orange", "-.")]:
        ax.axhline(val, color=col, ls=ls, lw=1.5, label=f"{lab} = {val:.0f}")
    ax.set_ylim(0, top)
    ax.set_xticks(range(1, 10))
    ax.set_xlabel("Window pixels, sorted (order statistics)")
    ax.set_ylabel("Intensity")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_window_order_statistics.png"), dpi=100)
    plt.close(fig)
    # 2) one row across a clear edge of the blurred frame: smoothed, detail, sharpened
    g = cv2.GaussianBlur(clean, (0, 0), BLUR)
    cys, cxs = detail_window(clean)
    gx = cv2.Sobel(cv2.GaussianBlur(clean, (0, 0), 2), cv2.CV_64F, 1, 0)
    best, pick = -1, None
    for yy, xx in np.argwhere(np.abs(gx[cys, cxs]) > np.percentile(np.abs(gx[cys, cxs]), 99)):
        yy, xx = yy + cys.start, xx + cxs.start
        if xx < 10 or xx > clean.shape[1] - 11:
            continue
        prof = clean[yy, xx - 8:xx + 9]
        steps = np.diff(prof) * np.sign(prof[-1] - prof[0])
        score = abs(prof[-1] - prof[0]) * (steps > -3).mean()        # a big, one-way step
        if score > best:
            best, pick = score, (yy, xx)
    y, x = pick
    xs_ = np.arange(x - 8, x + 9)
    row = lambda im: im[y, xs_]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(xs_ - x, row(clean), color="0.6", lw=3, label="clean")
    ax.plot(xs_ - x, row(g), color="black", lw=1.5, marker="o", ms=3, label="blurred g")
    ax.plot(xs_ - x, row(midpoint(g)), color="tab:orange", ls="--", label="midpoint of g (smoother)")
    ax.plot(xs_ - x, row(g - midpoint(g)), color="tab:purple", lw=1, label="detail = g − midpoint")
    ax.plot(xs_ - x, row(unsharp(g, midpoint)), color="tab:red", lw=2, label="sharpened = g + detail")
    ax.set_xlabel(f"Pixels from the edge (row {y})")
    ax.set_ylabel("Intensity")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_sharpening_profile.png"), dpi=100)
    plt.close(fig)
    return (y, x)


# ---------- 10. Run folder ----------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder")
    p.add_argument("--mean", type=float, default=MEAN)
    p.add_argument("--var", type=float, default=VAR)
    p.add_argument("--sp", type=float, default=SP)
    args = p.parse_args()

    exts = (".png", ".jpg", ".jpeg", ".bmp")
    files = sorted(f for f in os.listdir(args.folder) if f.lower().endswith(exts))
    out_dir = os.path.join("lecture5_results", f"var_{args.var:g}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== Set: mean {args.mean}, variance {args.var:g}, salt & pepper density {args.sp} ===")
    all_errors, summary = [], {}
    for f in files:
        name, est, errors = process(os.path.join(args.folder, f), out_dir, args.mean, args.var, args.sp)
        all_errors.append(errors)
        summary[name] = {"estimate": {k: {"type": e["type"], "var": e["var"], "params": {a: float(b) for a, b in e["params"].items()},
                                          "points": e["points"], "slope": e["slope"]}
                                      for k, e in est.items()}, "errors": errors}
        for k, e in est.items():
            par = ", ".join(f"{a} {b:.3g}" for a, b in e["params"].items())
            print(f"{name:4s} {k:14s} -> {e['type']:14s} var {e['var']:7.1f}   {par}")
    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(summary, fh, indent=1)

    # MSE (RMSE), offset removed, edges, sharpness: mean over the images
    print("\nMean over images: MSE (RMSE) | MSE no offset | MSE on edges | edge sharpness vs clean | runtime")
    for k in all_errors[0]:
        print(f"\n{k}")
        for m in all_errors[0][k]:
            v = np.mean([e[k][m] for e in all_errors], axis=0)
            line = f"  {m:26s} {v[0]:8.1f} ({np.sqrt(v[0]):5.1f}) | {v[1]:7.1f}"
            if len(v) > 2:
                line += f" | {v[2]:7.1f} | {v[3]:.2f}"
            if len(v) > 4:
                line += f" | {v[4]:.2f} s"
            print(line)
