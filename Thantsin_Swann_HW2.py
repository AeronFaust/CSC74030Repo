# __define-ocg__
# Logarithmic Image Processing: LIP (Jourlin & Pinoli, 1988) and
# Parameterized LIP (Panetta, Wharton & Agaian, 2008)
# Usage:
#   python lip_simple.py frames/            -> pairs consecutive frames (1&2, 2&3, ...)
#   python lip_simple.py gt/ recon/         -> pairs files with the same name

import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

M = 256.0     # LIP upper bound (8-bit images)
C = 0.5       # scalar for c ⊗ a (c < 1 brightens, c > 1 darkens)

# PLIP parameters: functions of M replace M; beta controls phi
GAMMA = 1026.0   # gamma(M), used by addition and scalar multiplication
K     = 1026.0   # k(M), used by subtraction
LAM   = 1026.0   # lambda(M), used by phi
BETA  = 2.0      # beta = 1 and GAMMA = K = LAM = M gives back LIP


# ---------- 1. Convert between intensity and gray tone ----------
# gray tone g(i,j) = M - I(i,j): 0 = white, M = black
def to_tone(img):
    return M - img.astype(np.float64)

def to_image(tone):
    return np.clip(M - tone, 0, 255).astype(np.uint8)


# ---------- 2. LIP operations ----------
def phi(a):
    # phi(a) = -M ln(1 - a/M)
    return -M * np.log(np.clip(1 - a / M, 1e-6, None))

def phi_inv(x):
    # phi_inv(a) = M (1 - exp(-a/M))
    return M * (1 - np.exp(-x / M))

def lip_add(a, b):
    # a ⊕ b = a + b - ab/M  (never exceeds M)
    return a + b - a * b / M

def lip_sub(a, b):
    # a ⊖ b = M(a - b)/(M - b)  (difference scaled by brightness of b)
    return M * (a - b) / np.clip(M - b, 1e-6, None)

def lip_scalar(c, a):
    # c ⊗ a = M - M(1 - a/M)^c
    return M - M * (1 - a / M) ** c

def lip_mul(a, b):
    # a ⊗ b = phi_inv(phi(a) * phi(b))
    return phi_inv(phi(a) * phi(b))

def lip_div(a, b):
    # a ⊘ b = phi_inv(phi(a) / phi(b))
    return phi_inv(phi(a) / np.clip(phi(b), 1e-6, None))


# ---------- 3. PLIP operations ----------
def pphi(a):
    # phi(a) = -lambda(M) ln^beta(1 - a/lambda(M))
    return LAM * np.abs(np.log(np.clip(1 - a / LAM, 1e-6, None))) ** BETA

def pphi_inv(x):
    # phi_inv(a) = lambda(M) [1 - exp(-(a/lambda(M))^(1/beta))]
    return LAM * (1 - np.exp(-(np.abs(x) / LAM) ** (1 / BETA)))

def plip_add(a, b):
    # a ⊕ b = a + b - ab/gamma(M)
    return a + b - a * b / GAMMA

def plip_sub(a, b):
    # a ⊖ b = k(M)(a - b)/(k(M) - b)
    return K * (a - b) / np.clip(K - b, 1e-6, None)

def plip_scalar(c, a):
    # c ⊗ a = gamma(M) - gamma(M)(1 - a/gamma(M))^c
    return GAMMA - GAMMA * (1 - a / GAMMA) ** c

def plip_mul(a, b):
    # a ⊗ b = phi_inv(phi(a) * phi(b))
    return pphi_inv(pphi(a) * pphi(b))

def plip_div(a, b):
    # a ⊘ b = phi_inv(phi(a) / phi(b))
    return pphi_inv(pphi(a) / np.clip(pphi(b), 1e-6, None))


# ---------- 4. Classical (regular) arithmetic for comparison ----------
def cls_add(a, b):    return np.clip(a.astype(float) + b, 0, 255).astype(np.uint8)
def cls_sub(a, b):    return np.abs(a.astype(float) - b).astype(np.uint8)  # difference map
def cls_scalar(c, a): return np.clip(a.astype(float) / c, 0, 255).astype(np.uint8)  # same direction as c ⊗ a
def cls_mul(a, b):    return np.clip(a.astype(float) * b / 255, 0, 255).astype(np.uint8)
def cls_div(a, b):    return np.clip(255 * a.astype(float) / np.clip(b.astype(float), 1, None), 0, 255).astype(np.uint8)


# ---------- 5. Check PLIP algebraic properties on random gray tones ----------
def check_properties():
    rng = np.random.default_rng(0)
    f, g, h = rng.uniform(0, 250, (3, 1000))
    c, d = 0.7, 1.8
    checks = {
        "Commutativity   f ⊕ g = g ⊕ f":            (plip_add(f, g), plip_add(g, f)),
        "Associativity   (f ⊕ g) ⊕ h = f ⊕ (g ⊕ h)": (plip_add(plip_add(f, g), h), plip_add(f, plip_add(g, h))),
        "Unit element    f ⊕ 0 = f":                (plip_add(f, 0), f),
        "Distributivity  (c+d) ⊗ f = c⊗f ⊕ d⊗f":     (plip_scalar(c + d, f), plip_add(plip_scalar(c, f), plip_scalar(d, f))),
        "Commutativity   f ⊗ g = g ⊗ f":            (plip_mul(f, g), plip_mul(g, f)),
    }
    print("PLIP property checks:")
    for name, (lhs, rhs) in checks.items():
        print(f"  {name:45s} {'holds' if np.allclose(lhs, rhs) else 'FAILS'}")


# ---------- 6. Process one pair of images ----------
def process_pair(a, b, name, out_dir):
    b = cv2.resize(b, (a.shape[1], a.shape[0]))  # match sizes
    fa, fb = to_tone(a), to_tone(b)

    # rows = operations, columns = Classical | LIP | PLIP
    results = {
        "add": (cls_add(a, b), to_image(lip_add(fa, fb)), to_image(plip_add(fa, fb))),
        "sub": (cls_sub(a, b),                                        # difference maps
                np.clip(np.abs(lip_sub(fa, fb)), 0, 255).astype(np.uint8),
                np.clip(np.abs(plip_sub(fa, fb)), 0, 255).astype(np.uint8)),
        "scalar": (cls_scalar(C, a), to_image(lip_scalar(C, fa)), to_image(plip_scalar(C, fa))),
        "mul": (cls_mul(a, b), to_image(lip_mul(fa, fb)), to_image(plip_mul(fa, fb))),
        "div": (cls_div(a, b), to_image(lip_div(fa, fb)), to_image(plip_div(fa, fb))),
    }
    models = ["Classical", "LIP", "PLIP"]

    # save a 6 x 3 grid: inputs on top, one row per operation
    fig, axes = plt.subplots(6, 3, figsize=(10, 19))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (title, img) in zip(axes[0], [("Input A", a), ("Input B", b)]):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
    for r, (op, imgs) in enumerate(results.items(), start=1):
        for col, img in enumerate(imgs):
            axes[r, col].imshow(img, cmap="gray", vmin=0, vmax=255)
            axes[r, col].set_title(f"{models[col]} {op}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}.png"), dpi=90)
    plt.close(fig)


# ---------- 7. Build list of image pairs from folder(s) ----------
EXTS = (".png", ".jpg", ".jpeg", ".bmp")

def list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(EXTS))

def build_pairs(args):
    if len(args) == 1:
        # one folder: consecutive frames
        folder = args[0]
        files = list_images(folder)
        return [(os.path.join(folder, files[i]), os.path.join(folder, files[i + 1]))
                for i in range(len(files) - 1)]
    # two folders: same filename in each
    common = sorted(set(list_images(args[0])) & set(list_images(args[1])))
    return [(os.path.join(args[0], f), os.path.join(args[1], f)) for f in common]


# ---------- 8. Run: property checks, then every pair ----------
if __name__ == "__main__":
    check_properties()

    pairs = build_pairs(sys.argv[1:])
    print(f"\nFound {len(pairs)} image pairs")
    out_dir = "lip_results"
    os.makedirs(out_dir, exist_ok=True)

    for pa, pb in pairs:
        a = cv2.imread(pa, cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(pb, cv2.IMREAD_GRAYSCALE)
        name = os.path.splitext(os.path.basename(pa))[0]
        process_pair(a, b, name, out_dir)
        print("done:", name)
