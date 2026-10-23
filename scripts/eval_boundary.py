#!/usr/bin/env python3
"""
Boundary-restricted PSNR / SSIM for TNT renders.

Protocol
--------
Boundary masks are extracted ONCE from the ground-truth images of each scene and
shared by every method, so no method is ever evaluated on a mask derived from its
own output. GT is read from whichever experiment directory is discovered first
for that (scene, split) -- every experiment's gt/ is a copy of the same ground
truth, so which one is picked does not matter.

Band construction follows the trimap protocol of Kohli et al. (IJCV 2009) and
Chen et al. (DeepLab, PAMI 2018), with band radii swept as in Pohlen et al.
(FRRN).  Band widths are given in ABSOLUTE PIXELS rather than as a fraction of
the image diagonal, following the pixel-distance consideration of Cheng et al.
(Boundary IoU, CVPR 2021), so that scenes of differing resolution remain
comparable.

Edge detection:
  --edge sobel  (default)  gradient magnitude thresholded at a fixed percentile.
                           No literature default; the protocol is self-contained
                           because mask coverage is reported alongside every score.
  --edge canny             OpenCV Canny.  Thresholds come from a widely used
                           median heuristic that has NO published source; use
                           only for sanity checks, not for reported numbers.

Both boundary and non-boundary (complement) scores are reported: a targeted
improvement shows up as a gain inside the band with the complement unchanged.

Layout expected
---------------
{output_root}/{experiment}/TNT_GOF/TrainingSet/{scene}/{split}/{iteration}/{gt,renders}/*.png
as written by `scripts/run_experiments.py --dataset tnt_mesh` (render stage).
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

SUB = "TNT_GOF/TrainingSet"


# ----------------------------------------------------------------- edges
def detect_edges(gt_bgr, mode, percentile):
    """Return a boolean edge map (1-px-ish, before dilation)."""
    gray = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2GRAY)
    if mode == "canny":
        v = float(np.median(gray))
        lo = int(max(0, 0.67 * v))
        hi = int(min(255, 1.33 * v))
        return cv2.Canny(gray, lo, hi) > 0

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    return mag >= np.percentile(mag, percentile)


def band(edges, radius):
    if radius <= 0:
        return edges
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    return cv2.dilate(edges.astype(np.uint8), k).astype(bool)


# ---------------------------------------------------------------- metrics
def masked_psnr(gt, pred, mask):
    if not mask.any():
        return float("nan")
    d = (gt.astype(np.float64) - pred.astype(np.float64)) ** 2
    mse = d[mask].mean()
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def ssim_map(gt, pred):
    """Per-pixel SSIM map averaged over channels (Gaussian 11x11, sigma=1.5)."""
    from skimage.metrics import structural_similarity as ssim
    _, S = ssim(gt, pred, data_range=255, channel_axis=2, full=True,
                gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    return S.mean(axis=2)


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_root", required=True, type=Path, help="--output_root of run_experiments.py")
    ap.add_argument("--iteration", default="ours_30000")
    ap.add_argument("--split", default="test", choices=["train", "test", "*"])
    ap.add_argument("--experiments", nargs="*", default=None,
                    help="substring filters on the experiment dir name")
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--radii", type=int, nargs="*", default=[1, 2, 4, 8, 16])
    ap.add_argument("--edge", default="sobel", choices=["sobel", "canny"])
    ap.add_argument("--percentile", type=float, default=95.0)   # Table 3 of the paper: test split, 95th percentile
    ap.add_argument("--ssim", action="store_true", help="also compute SSIM (slower)")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <output_root>/boundary_tnt_p<percentile>.csv")
    ap.add_argument("--per-image-out", type=Path, default=None)
    a = ap.parse_args()

    # ---- discover render dirs, group by (scene, split) --------------------
    pattern = f"*/{SUB}/*/{a.split}/{a.iteration}/renders"
    dirs = sorted(d for d in a.output_root.glob(pattern) if d.is_dir())
    if a.experiments:
        dirs = [d for d in dirs
                if any(f in d.relative_to(a.output_root).parts[0] for f in a.experiments)]
    if a.scenes:
        dirs = [d for d in dirs if d.relative_to(a.output_root).parts[3] in a.scenes]
    if not dirs:
        print("no renders dir matched:", a.output_root / pattern)
        return

    groups = {}
    for d in dirs:
        p = d.relative_to(a.output_root).parts
        groups.setdefault((p[3], p[4]), []).append((p[0], d))   # (scene, split)

    per_image, agg = [], []

    # ---- one scene at a time: masks are built once and reused across methods
    for (scene, split), exps in tqdm(sorted(groups.items()), leave=True):
        # gt/ is identical across every experiment for this (scene, split);
        # no sorting needed, just use whichever experiment was discovered
        # first as the source, built directly from the known layout.
        ref_exp = exps[0][0]
        iter_dir = a.output_root / ref_exp / SUB / scene / split / a.iteration
        gdir = iter_dir / "gt"
        if not gdir.is_dir():
            print(f"[skip] no gt/ for {scene}/{split}")
            continue

        stems = sorted(p.stem for p in gdir.glob("*.png"))
        stems = [s for s in stems
                 if all((rd / f"{s}.png").exists() for _, rd in exps)]
        if not stems:
            print(f"[skip] no stem common to all methods in {scene}/{split}")
            continue

        buckets = {}                      # (exp, radius) -> [rec, ...]
        for stem in tqdm(stems, leave=False, desc=f"{scene} - {split}"):
            gt = cv2.imread(str(gdir / f"{stem}.png"), cv2.IMREAD_COLOR)
            if gt is None:
                print(f"[warn] unreadable gt {scene}/{stem}")
                continue

            # edges once per frame, masks once per (frame, radius)
            edges = detect_edges(gt, a.edge, a.percentile)
            masks = {r: band(edges, r) for r in a.radii}
            full = np.ones(gt.shape[:2], dtype=bool)

            for exp, rdir in exps:
                pr = cv2.imread(str(rdir / f"{stem}.png"), cv2.IMREAD_COLOR)
                if pr is None:
                    print(f"[warn] unreadable render {exp}/{scene}/{stem}")
                    continue
                if pr.shape != gt.shape:
                    pr = cv2.resize(pr, (gt.shape[1], gt.shape[0]),
                                    interpolation=cv2.INTER_AREA)
                    print(f"[warn] resized render {exp}/{scene}/{stem}")

                S = ssim_map(gt, pr) if a.ssim else None
                psnr_full = masked_psnr(gt, pr, full)
                ssim_full = float(S.mean()) if S is not None else None

                for r in a.radii:
                    m = masks[r]
                    inv = ~m
                    rec = dict(experiment=exp, scene=scene, split=split,
                               image=stem, radius=r,
                               coverage=float(m.mean()),
                               psnr_boundary=masked_psnr(gt, pr, m),
                               psnr_nonboundary=masked_psnr(gt, pr, inv),
                               psnr_full=psnr_full)
                    if S is not None:
                        rec["ssim_boundary"] = float(S[m].mean()) if m.any() else np.nan
                        rec["ssim_nonboundary"] = float(S[inv].mean()) if inv.any() else np.nan
                        rec["ssim_full"] = ssim_full
                    per_image.append(rec)
                    buckets.setdefault((exp, r), []).append(rec)

            del masks, edges          # keep peak memory to one frame

        for (exp, r), recs in sorted(buckets.items()):
            keys = [k for k in recs[0]
                    if k.startswith(("psnr", "ssim")) or k == "coverage"]
            agg.append(dict(experiment=exp, scene=scene, split=split, radius=r,
                            n_images=len(recs),
                            **{k: float(np.nanmean([x[k] for x in recs]))
                               for k in keys}))
        print(f"[ok] {scene}/{split}: {len(stems)} images x {len(exps)} methods")

    # ---- write ------------------------------------------------------------
    if agg:
        a.out = a.out or a.output_root / f"boundary_tnt_p{int(a.percentile):02d}.csv"
        with a.out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(agg[0].keys()))
            w.writeheader()
            w.writerows(agg)
        print(f"{len(agg)} rows -> {a.out}")
        # scene-averaged boundary PSNR per experiment, laid out as Table 3 of the paper
        print("\nexperiment\t" + "\t".join(f"r={r}" for r in a.radii) + "\tfull")
        for exp in sorted({row["experiment"] for row in agg}):
            mean = lambda k, r: np.mean([row[k] for row in agg if row["experiment"] == exp and row["radius"] == r])
            print(exp + "\t" + "\t".join(f"{mean('psnr_boundary', r):.2f}" for r in a.radii)
                  + f"\t{mean('psnr_full', a.radii[0]):.2f}")
    if a.per_image_out and per_image:
        with a.per_image_out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(per_image[0].keys()))
            w.writeheader()
            w.writerows(per_image)
        print(f"{len(per_image)} rows -> {a.per_image_out}")


if __name__ == "__main__":
    main()