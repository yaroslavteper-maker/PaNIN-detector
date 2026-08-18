"""
Modernized, CPU-only, single-image VISTA / MicePan inference.

Reproduces the essential pipeline from ProcessImages.py without the CUDA / TF1 /
staintools stack:
  1. Fit Reinhard normalizer to the published target image.
  2. Load an input H&E image (optionally center-crop for a quick run).
  3. Luminosity-standardize + Reinhard-normalize the region (non-white pixels).
  4. Tile into 512x512, predict all 3 UNets, stitch probability maps.
  5. Threshold (meta 0.5 / neo 0.7 / normal 0.3), combine with priority, derive
     stroma from whitespace, and write a combined color mask + per-class masks.

Usage:
  python run_vista.py INPUT.tif [--crop N] [--out DIR]
"""
import os, sys, time, argparse
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize
from vista_norm import luminosity_standardize, ReinhardColorNormalizer, tissue_mask
from vista_model import load_unet, MODELS

TILE = 512
COLORS = {  # RGB, matching ProcessImages.py
    "neoplasia":  (230, 210, 30),
    "metaplasia": (222, 31, 123),
    "normal":     (122, 230, 213),
    "stroma":     (19, 16, 163),
}


def center_crop(img, n):
    if n <= 0:
        return img
    h, w = img.shape[:2]
    n = min(n, h, w)
    y = (h - n) // 2
    x = (w - n) // 2
    return img[y:y + n, x:x + n]


def fit_normalizer(target_path):
    target = imread(target_path)[:, :, :3]
    target = luminosity_standardize(target)
    norm = ReinhardColorNormalizer()
    norm.fit(target, tissue_mask(target))
    return norm


def normalize_region(img, norm):
    mask = tissue_mask(img)
    std = luminosity_standardize(img)
    out = norm.transform(std, mask)
    out[~mask] = img[~mask]  # leave white background untouched
    return out


def predict_maps(region, models):
    """Return dict class -> float prob map over the region (512 tiling)."""
    h, w = region.shape[:2]
    ph = (-h) % TILE
    pw = (-w) % TILE
    padded = np.pad(region, ((0, ph), (0, pw), (0, 0)), mode="constant", constant_values=255)
    H, W = padded.shape[:2]
    maps = {k: np.zeros((H, W), np.float32) for k in models}
    ntiles = (H // TILE) * (W // TILE)
    done = 0
    for y in range(0, H, TILE):
        for x in range(0, W, TILE):
            block = padded[y:y + TILE, x:x + TILE].astype(np.float32)[None]
            for k, (model, _) in models.items():
                pred = model.predict(block, verbose=0)
                maps[k][y:y + TILE, x:x + TILE] = np.squeeze(pred)
            done += 1
            sys.stdout.write(f"\r  tiles {done}/{ntiles}")
            sys.stdout.flush()
    print()
    return {k: v[:h, :w] for k, v in maps.items()}


def combine(maps, region):
    meta = maps["metaplasia"] >= MODELS["metaplasia"][1]
    neo = maps["neoplasia"] >= MODELS["neoplasia"][1]
    norm_ = maps["normal"] >= MODELS["normal"][1]
    white = (region[:, :, 0] >= 200) & (region[:, :, 1] >= 200) & (region[:, :, 2] >= 200)
    for m in (meta, neo, norm_):
        m[white] = False
    # priority matches ProcessImages.py: normal > metaplasia > neoplasia
    neo[meta] = False
    neo[norm_] = False
    meta[norm_] = False
    stroma = ~white & ~neo & ~meta & ~norm_
    out = np.zeros_like(region)
    out[neo] = COLORS["neoplasia"]
    out[meta] = COLORS["metaplasia"]
    out[norm_] = COLORS["normal"]
    out[stroma] = COLORS["stroma"]
    frac = {
        "neoplasia": float(neo.mean()),
        "metaplasia": float(meta.mean()),
        "normal": float(norm_.mean()),
        "stroma": float(stroma.mean()),
        "white": float(white.mean()),
    }
    return out, frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--crop", type=int, default=2048, help="center-crop side (0 = full image)")
    ap.add_argument("--out", default="out")
    ap.add_argument("--target", default="../norm/TargetForNormalization-Copy1.tif")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    print("Fitting normalizer to target...")
    norm = fit_normalizer(args.target)
    print("  target LAB means:", tuple(round(m, 3) for m in norm.target_means))
    print("  target LAB stds :", tuple(round(s, 3) for s in norm.target_stds))

    print(f"Loading {args.input} ...")
    img = imread(args.input)[:, :, :3]
    img = center_crop(img, args.crop)
    print("  region:", img.shape)

    print("Normalizing...")
    region = normalize_region(img, norm)

    print("Loading UNets...")
    models = {k: (load_unet(f), thr) for k, (f, thr) in MODELS.items()}

    print("Predicting...")
    maps = predict_maps(region, models)

    combined, frac = combine(maps, region)
    imsave(os.path.join(args.out, "region_normalized.png"), region.astype(np.uint8))
    imsave(os.path.join(args.out, "combined.png"), combined.astype(np.uint8))
    for k in ("neoplasia", "metaplasia", "normal"):
        imsave(os.path.join(args.out, f"prob_{k}.png"), (maps[k] * 255).astype(np.uint8))

    print("\nComposition (fraction of region):")
    for k, v in frac.items():
        print(f"  {k:11s} {v*100:5.1f}%")
    print(f"\nDone in {time.time()-t0:.1f}s -> {args.out}/")


if __name__ == "__main__":
    main()
