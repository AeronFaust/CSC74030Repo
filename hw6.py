# __define-ocg__
"""
Image enhancement with spatial (point) operations, scored by the EMEE quality measure.

Pipeline (per the assignment):
  1. Contrast stretching: linear map of [a, b] onto [0, 255] (min-max or percentile limits)
  2. Image quality measure EMEE: 2.1 filter function alpha r^alpha ln r, 2.2 its average over k1 x k2 blocks
  3. Histogram equalization (own implementation, checked against cv2.equalizeHist) + the u -> v mapping
  4. Gamma correction: g = 255 (f / 255)^gamma
  5. Color enhancement in HSV: enhance V (stretch / naive HE / HE, then gamma), median + linear gain on S,
     back to RGB, or color correction R_new = R_old V_new / V_old (formula 5)
  3.3 Bonus: spatial enhancement = gamma + unsharp masking + percentile stretch
"""
import os
import json
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- 0. io

def load_rgb(path):
    return cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float64)


def to_gray(rgb):
    """Y = 0.299 R + 0.587 G + 0.114 B (OpenCV)."""
    return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)


def to_u8(x):
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- 1. contrast stretching

def contrast_stretch(g, lo=0, hi=100):
    """v = 255 (u - a) / (b - a), clipped; a, b = lo-th and hi-th percentiles (0, 100 = min-max)."""
    a, b = np.percentile(g, [lo, hi])
    return np.clip((g - a) / max(b - a, 1e-6) * 255, 0, 255), (a, b)


# ---------------------------------------------------------------- 2. EMEE

def emee_filter(r, alpha=0.5):
    """2.1  alpha r^alpha ln r,  r = I_max / (I_min + c);  -> 0 as r -> 0."""
    r = np.asarray(r, np.float64)
    out = np.zeros_like(r)
    pos = r > 0
    out[pos] = alpha * r[pos] ** alpha * np.log(r[pos])
    return out


def block_ratios(g, block=8, c=1.0):
    """I_max;k,l / (I_min;k,l + c) for every block (k, l) of block x block pixels."""
    H, W = g.shape
    H, W = H - H % block, W - W % block
    blk = g[:H, :W].reshape(H // block, block, W // block, block).swapaxes(1, 2)
    return blk.max(axis=(2, 3)) / (blk.min(axis=(2, 3)) + c)


def emee_map(g, alpha=0.5, block=8, c=1.0):
    """Filter function per block (k1 x k2 map)."""
    return emee_filter(block_ratios(g, block, c), alpha)


def emee(g, alpha=0.5, block=8, c=1.0):
    """2.2  EMEE = 1 / (k1 k2) sum_l sum_k alpha (I_max / (I_min + c))^alpha ln(I_max / (I_min + c))."""
    return float(emee_map(g, alpha, block, c).mean())


# ---------------------------------------------------------------- 3. histogram equalization

def histogram(g):
    """h(r_k) = number of pixels at level r_k, k = 0..255."""
    return np.bincount(to_u8(g).ravel(), minlength=256)


def equalize_lut(g, naive=False):
    """v = T(u) = 255 (CDF(u) - CDF_min) / (N - CDF_min);  naive: v = 255 CDF(u) / N."""
    cdf = histogram(g).cumsum()
    if naive:
        return np.round(255 * cdf / cdf[-1])
    cmin = cdf[cdf > 0][0]
    return np.round((cdf - cmin) / (cdf[-1] - cmin) * 255)


def equalize(g, naive=False):
    """Apply T to every pixel; returns (image, T)."""
    lut = equalize_lut(g, naive)
    return lut[to_u8(g)], lut


# ---------------------------------------------------------------- 4. gamma correction

def gamma_correct(f, gamma):
    """g = 255 (f / 255)^gamma;  gamma < 1 brightens, gamma > 1 darkens."""
    return 255 * (np.clip(f, 0, 255) / 255) ** gamma


def gamma_to_mean(f, target_mean):
    """gamma that maps mean(f) back to target_mean: (m / 255)^gamma = target / 255."""
    m = np.clip(f.mean() / 255, 1e-3, 0.999)
    return float(np.log(target_mean / 255) / np.log(m))


# ---------------------------------------------------------------- 5. color enhancement in HSV

def rgb_to_hsv(rgb):
    """H in degrees (0-360), S and V in 0-1."""
    return cv2.cvtColor((rgb / 255).astype(np.float32), cv2.COLOR_RGB2HSV).astype(np.float64)


def hsv_to_rgb(hsv):
    return cv2.cvtColor(hsv.astype(np.float32), cv2.COLOR_HSV2RGB).astype(np.float64) * 255


# V enhancement methods (each followed by gamma correction back to the original mean)
METHODS = {
    "Linear stretching + gamma": lambda v, pct: contrast_stretch(v, *pct)[0],
    "Naive HE + gamma":          lambda v, pct: equalize(v, naive=True)[0],
    "HE + gamma":                lambda v, pct: equalize(v)[0],
}


def enhance_saturation(s, gain=1.3, ksize=3):
    """S' = gain * median_k(S), clipped to [0, 1] (median removes isolated color specks)."""
    s = cv2.medianBlur((s * 255).astype(np.uint8), ksize).astype(np.float64) / 255
    return np.clip(gain * s, 0, 1)


def enhance_color(rgb, method, pct=(1, 99), s_gain=1.3, s_ksize=3):
    """Steps 1-4: RGB -> HSV, V' = gamma(method(V)), S' = gain * median(S), H S' V' -> RGB ("early");
    formula 5: R_new = R_old V_new / V_old (same for G, B) ("corrected", keeps the input's color ratios)."""
    hsv = rgb_to_hsv(rgb)
    v = hsv[..., 2] * 255
    v_new = METHODS[method](v, pct)
    gam = gamma_to_mean(v_new, v.mean())
    v_new = gamma_correct(v_new, gam)
    s_new = enhance_saturation(hsv[..., 1], s_gain, s_ksize)
    early = hsv_to_rgb(np.stack([hsv[..., 0], s_new, v_new / 255], axis=-1))
    corrected = np.clip(rgb * (v_new / np.maximum(v, 1))[..., None], 0, 255)
    return dict(early=early, corrected=corrected, gamma=gam, v_new=v_new, s_new=s_new)


# ---------------------------------------------------------------- 3.3 bonus: spatial enhancement

def spatial_enhance(g, gamma=0.8, sigma=1.5, amount=0.8, pct=(1, 99)):
    """1) lift shadows: gamma;  2) unsharp mask: s = f + amount (f - G_sigma * f);  3) percentile stretch."""
    lifted = gamma_correct(g, gamma)
    sharp = np.clip(lifted + amount * (lifted - cv2.GaussianBlur(lifted, (0, 0), sigma)), 0, 255)
    out, _ = contrast_stretch(sharp, *pct)
    return {f"Gamma {gamma:g}": lifted, "+ unsharp mask": sharp, "+ stretch (final)": out}


# ---------------------------------------------------------------- 6. figures

def detail_window(gray, size=220):
    """size x size window with the most edge energy (usually face and hands)."""
    gm = np.hypot(cv2.Sobel(gray, cv2.CV_64F, 1, 0), cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    score = cv2.boxFilter(gm, -1, (size, size))
    h, w = gray.shape
    score[: size // 2, :] = score[h - size // 2:, :] = 0
    score[:, : size // 2] = score[:, w - size // 2:] = 0
    cy, cx = np.unravel_index(score.argmax(), score.shape)
    return slice(cy - size // 2, cy + size // 2), slice(cx - size // 2, cx + size // 2)


def image_hist_grid(ims, path, cfg, color=False):
    """Each column: image (EMEE in title) over its histogram (R, G, B rows for color)."""
    rows = 1 + (3 if color else 1)
    fig, axes = plt.subplots(rows, len(ims), figsize=(3.6 * len(ims), 3.2 + 1.5 * (rows - 1)),
                             gridspec_kw={"height_ratios": [2.2] + [1] * (rows - 1)})
    for c, (title, im) in enumerate(ims.items()):
        g = im if im.ndim == 2 else to_gray(im)
        axes[0, c].imshow(to_u8(im), cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=255)
        axes[0, c].set_title(f"{title}\nEMEE = {emee(g, cfg.alpha, cfg.block, cfg.c):.3f}", fontsize=10)
        axes[0, c].axis("off")
        chans = [(im[..., k], col) for k, col in enumerate(["red", "green", "blue"])] if color else [(im, "0.4")]
        for r, (ch, col) in enumerate(chans, start=1):
            axes[r, c].bar(np.arange(256), histogram(ch), width=1, color=col)
            axes[r, c].set_xlim(0, 255)
            axes[r, c].set_yticks([])
            axes[r, c].tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=80)
    plt.close(fig)


def zoom_figure(ims, path):
    """Full image with two boxes (most detailed region, red; next one, gold) and both crops underneath."""
    first = next(iter(ims.values()))
    g = first if first.ndim == 2 else first.mean(axis=-1)
    ys, xs = detail_window(g, 160)
    boxes = [(ys, xs, "red")]
    gm = cv2.boxFilter(np.hypot(cv2.Sobel(g, cv2.CV_64F, 1, 0), cv2.Sobel(g, cv2.CV_64F, 0, 1)), -1, (120, 120))
    gm[max(ys.start - 120, 0):ys.stop + 120, max(xs.start - 120, 0):xs.stop + 120] = 0
    gm[:60], gm[-60:], gm[:, :60], gm[:, -60:] = 0, 0, 0, 0
    cy, cx = np.unravel_index(gm.argmax(), gm.shape)
    boxes.append((slice(cy - 60, cy + 60), slice(cx - 60, cx + 60), "gold"))
    fig, axes = plt.subplots(2, len(ims), figsize=(3.2 * len(ims), 4.9), gridspec_kw={"height_ratios": [2, 1]})
    for c, (title, im) in enumerate(ims.items()):
        cmap = "gray" if im.ndim == 2 else None
        axes[0, c].imshow(to_u8(im), cmap=cmap, vmin=0, vmax=255)
        for by, bx, col in boxes:
            axes[0, c].add_patch(plt.Rectangle((bx.start, by.start), bx.stop - bx.start, by.stop - by.start,
                                               fill=False, ec=col, lw=2))
        axes[0, c].set_title(title, fontsize=10)
        crops = [cv2.resize(to_u8(im[by, bx]), (150, 150), interpolation=cv2.INTER_NEAREST) for by, bx, _ in boxes]
        axes[1, c].imshow(np.concatenate(crops, axis=1), cmap=cmap, vmin=0, vmax=255)
        axes[1, c].add_patch(plt.Rectangle((0, 0), 149, 149, fill=False, ec="red", lw=3))
        axes[1, c].add_patch(plt.Rectangle((150, 0), 149, 149, fill=False, ec="gold", lw=3))
        for ax in axes[:, c]:
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=80)
    plt.close(fig)


def plot_emee_filter(path, alphas):
    """2.1: the filter function for several alphas."""
    r = np.linspace(0.05, 30, 400)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for a in alphas:
        ax.plot(r, emee_filter(r, a), label=f"alpha = {a:g}")
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_xlabel("r = I_max / (I_min + c)")
    ax.set_ylabel("alpha r^alpha ln r")
    ax.set_ylim(-1, 60)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=90)
    plt.close(fig)


def plot_gamma_curves(path, gammas):
    """4: g = 255 (f / 255)^gamma for each gamma."""
    u = np.arange(256)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for gm in gammas:
        ax.plot(u, gamma_correct(u, gm), label=f"gamma = {gm:g}")
    ax.plot(u, u, color="0.7", ls=":", label="gamma = 1 (identity)")
    ax.set_xlabel("input level f"); ax.set_ylabel("output level g")
    ax.set_xlim(0, 255); ax.set_ylim(0, 255)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=90)
    plt.close(fig)


def plot_transform(path, g, lut, stretch_lut, pct):
    """3.1.1: intensity transformation u -> v, input histogram behind it."""
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot(np.arange(256), lut, color="black", lw=2, label="Histogram equalization")
    ax.plot(np.arange(256), stretch_lut, color="tab:orange", ls="--", label=f"Stretch {pct[0]}-{pct[1]}%")
    ax.plot([0, 255], [0, 255], color="0.7", ls=":", label="identity")
    ax2 = ax.twinx()
    ax2.bar(np.arange(256), histogram(g), width=1, color="tab:blue", alpha=0.25)
    ax2.set_yticks([])
    ax.set_xlabel("input level u"); ax.set_ylabel("output level v")
    ax.set_xlim(0, 255); ax.set_ylim(0, 255)
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=90)
    plt.close(fig)


def plot_emee_map(path, g, cfg):
    """2.2: per-block EMEE next to the image (color scale capped at the 99th percentile)."""
    em = emee_map(g, cfg.alpha, cfg.block, cfg.c)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(g, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original")
    im = axes[1].imshow(em, cmap="magma", vmin=0, vmax=np.percentile(em, 99))
    axes[1].set_title(f"EMEE per {cfg.block} x {cfg.block} block (mean = {em.mean():.3f})")
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=80)
    plt.close(fig)


# ---------------------------------------------------------------- 7. run one image

def enhance_image(path, out_dir, cfg):
    """All parts on one image; saves figures, returns (name, results)."""
    name = os.path.splitext(os.path.basename(path))[0]
    out = lambda tag: os.path.join(out_dir, f"{name}_{tag}.png")
    score = lambda im: emee(im, cfg.alpha, cfg.block, cfg.c)
    pct = tuple(cfg.stretch)
    rgb = load_rgb(path)
    g = to_gray(rgb)
    res = {"emee": {}, "stats": {}}

    def rec(k, im):
        res["emee"][k] = score(im)
        res["stats"][k] = (float(im.mean()), float(im.std()))
    rec("Original", g)

    # 1. contrast stretching
    mm, ab_mm = contrast_stretch(g)
    pc, ab_pc = contrast_stretch(g, *pct)
    rec("Stretch min-max", mm); rec("Stretch 1-99%", pc)
    res["stretch_limits"] = {"min-max": ab_mm, "1-99%": ab_pc}
    image_hist_grid({"Original": g, "Stretch, min-max": mm, f"Stretch, {pct[0]}-{pct[1]}%": pc}, out("stretch"), cfg)

    # 2. EMEE
    plot_emee_map(out("emee_map"), g, cfg)
    res["emee_alpha"] = {str(a): emee(g, a, cfg.block, cfg.c) for a in cfg.alphas}

    # 3. histogram equalization
    he, lut = equalize(g)
    nhe, nlut = equalize(g, naive=True)
    res["he_matches_opencv"] = bool(np.array_equal(to_u8(he), cv2.equalizeHist(to_u8(g))))
    rec("Histogram equalization", he); rec("Naive HE", nhe)
    image_hist_grid({"Original": g, "Histogram equalization": he, f"Stretch, {pct[0]}-{pct[1]}%": pc}, out("equalize"), cfg)
    stretch_lut = np.clip((np.arange(256) - ab_pc[0]) / (ab_pc[1] - ab_pc[0]) * 255, 0, 255)
    res.update(lut=lut.tolist(), naive_lut=nlut.tolist(), stretch_lut=stretch_lut.tolist())
    plot_transform(out("transform"), g, lut, stretch_lut, pct)

    # 4. gamma correction
    gams = {f"gamma = {gm:g}": gamma_correct(g, gm) for gm in cfg.gammas}
    for k, v in gams.items():
        rec(k.capitalize(), v)
    image_hist_grid({"Original": g, **gams}, out("gamma"), cfg)

    # 5. color enhancement in HSV
    color = {m: enhance_color(rgb, m, pct, cfg.s_gain, cfg.s_ksize) for m in METHODS}
    res["color"] = {m: {"gamma": o["gamma"], "emee_early": score(to_gray(o["early"])),
                        "emee_corrected": score(to_gray(o["corrected"])),
                        "mean_S": float(rgb_to_hsv(o["early"])[..., 1].mean())} for m, o in color.items()}
    res["color"]["Input"] = {"emee": score(g), "mean_S": float(rgb_to_hsv(rgb)[..., 1].mean())}
    image_hist_grid({"Input image (RGB)": rgb, **{m: o["corrected"] for m, o in color.items()}}, out("color"), cfg, color=True)
    image_hist_grid({"Input image (RGB)": rgb, "HE + gamma, H S' V' -> RGB": color["HE + gamma"]["early"],
                     "HE + gamma, formula 5": color["HE + gamma"]["corrected"]}, out("color_steps"), cfg, color=True)
    zoom_figure({"Input": rgb, **{m: o["corrected"] for m, o in color.items()}}, out("color_zoom"))

    # 3.3 bonus: spatial enhancement
    steps = spatial_enhance(g, pct=pct)
    for k, v in steps.items():
        rec("Spatial: " + k, v)
    image_hist_grid({"Before": g, **steps}, out("spatial"), cfg)
    zoom_figure({"Before": g, "After": steps["+ stretch (final)"]}, out("spatial_zoom"))
    return name, res


def report(name, res, cfg):
    print(f"\n{name}: EMEE (alpha {cfg.alpha:g}, c {cfg.c:g}, {cfg.block} x {cfg.block} blocks) | mean | std")
    for k, v in res["emee"].items():
        print(f"  {k:28s} {v:7.3f} | {res['stats'][k][0]:6.1f} | {res['stats'][k][1]:5.1f}")
    print(f"  own HE equals cv2.equalizeHist: {res['he_matches_opencv']}")
    for m, d in res["color"].items():
        print(f"  color {m:26s} " + ", ".join(f"{a} {b:.3f}" for a, b in d.items()))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Image enhancement scored by EMEE")
    p.add_argument("images", help="folder of images (or one image)")
    p.add_argument("-o", "--out", default="enhancement_results")
    p.add_argument("--alpha", type=float, default=0.5, help="EMEE alpha")
    p.add_argument("--block", type=int, default=8, help="EMEE block size (k1 x k2 blocks of block x block)")
    p.add_argument("--c", type=float, default=1.0, help="EMEE constant c")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0], help="alphas for the filter-function plot")
    p.add_argument("--stretch", type=float, nargs=2, default=[1, 99], help="percentile limits for contrast stretching")
    p.add_argument("--gammas", type=float, nargs="+", default=[0.4, 0.67, 1.5, 2.5])
    p.add_argument("--s-gain", type=float, default=1.3, help="saturation gain in the HSV method")
    p.add_argument("--s-ksize", type=int, default=3, help="median kernel for the saturation channel")
    a = p.parse_args()

    files = ([os.path.join(a.images, f) for f in sorted(os.listdir(a.images))
              if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))] if os.path.isdir(a.images) else [a.images])
    os.makedirs(a.out, exist_ok=True)
    plot_emee_filter(os.path.join(a.out, "emee_filter.png"), a.alphas)
    plot_gamma_curves(os.path.join(a.out, "gamma_curves.png"), a.gammas)

    summary = {}
    for f in files:
        name, res = enhance_image(f, a.out, a)
        summary[name] = res
        report(name, res, a)
    with open(os.path.join(a.out, "results.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
