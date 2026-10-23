"""
2D histograms of the learned interior control point (c1, d1) per axis (Figure 4 of the paper).

    python scripts/plot_control_points.py --output_root output --experiments "pwl*" "pchip*"

Reads <output_root>/<experiment>/Mip-NeRF360/360_v2/<scene>/point_cloud/iteration_<N>/point_cloud.ply
and writes <output_root>/figures/control_points/<experiment>/<scene>_{u,v}.pdf, one figure per warped axis.
The kernel is taken from the model's cfg_args; pass --kernel for models trained before it was recorded.
"""

from argparse import ArgumentParser, Namespace  # Namespace is needed to eval cfg_args
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from plyfile import PlyData

DOF_MARKERS = "xyzabcdefg"   # attribute naming of GaussianModel.save_ply: inter_point_<marker><axis>
QMAX = 3.0                   # slope-ratio bound of the sr kernels (PWL_QMAX / PCHIP_QMAX)


def single_knot(p):
    """pwl, pchip: (c1, d1) = 2 * sigmoid(a, b) - 1."""
    return p * 2 - 1


def slope_ratio(p, eps=1e-3):
    """pwl_sr, pchip_sr: c1 = 2 sigmoid(a) - 1, R = exp(QMAX (2 sigmoid(b) - 1)), d1 from Eq. (14)."""
    u, v = np.clip(p[..., 0], eps, 1 - eps), np.clip(p[..., 1], eps, 1 - eps)
    c1 = 2 * u - 1
    R = np.exp(QMAX * (2 * v - 1))
    h0, h1 = c1 + 1, 1 - c1
    d1 = -1 + 2 * R * h0 / (h1 + R * h0)
    return np.stack([c1, d1], axis=-1)


CONTROL_POINT = {"pwl": single_knot, "pchip": single_knot, "pwl_sr": slope_ratio, "pchip_sr": slope_ratio}


def load_inter_points(ply_path):
    """Return the activated inter points, shape (N, axes, dof)."""
    props = PlyData.read(str(ply_path)).elements[0]
    names = [p.name for p in props.properties if p.name.startswith("inter_point_")]
    axes = 1 + max(int(n[-1]) for n in names)
    dof = len({n[-2] for n in names})
    raw = np.stack([np.stack([np.asarray(props[f"inter_point_{DOF_MARKERS[d]}{a}"]) for d in range(dof)], -1)
                    for a in range(axes)], axis=1)
    return 1 / (1 + np.exp(-raw))


def read_kernel(model_dir):
    cfg = model_dir / "cfg_args"
    return getattr(eval(cfg.read_text()), "kernel", None) if cfg.exists() else None


def plot(points, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    _, _, _, image = ax.hist2d(points[:, 0], points[:, 1], bins=200, range=[[-1, 1], [-1, 1]],
                               cmap="Reds", norm=LogNorm(), cmin=1, rasterized=True)
    image.set_alpha(0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    parser = ArgumentParser(description="Plot learned control point distributions (Figure 4)")
    parser.add_argument("--output_root", required=True, type=Path, help="--output_root of run_experiments.py")
    parser.add_argument("--experiments", nargs="+", default=["*"], help="globs over experiment folders")
    parser.add_argument("--scenes", nargs="+", default=["*"], help="globs over scene folders, e.g. 'bonsai_8'")
    parser.add_argument("--kernel", choices=list(CONTROL_POINT), default=None,
                        help="override the kernel recorded in cfg_args")
    parser.add_argument("--iteration", type=int, default=30_000)
    parser.add_argument("--out", type=Path, default=None, help="default: <output_root>/figures/control_points")
    args = parser.parse_args()
    out_root = args.out or args.output_root / "figures/control_points"

    model_dirs = sorted({d for e in args.experiments for s in args.scenes
                         for d in args.output_root.glob(f"{e}/Mip-NeRF360/360_v2/{s}")})
    for model_dir in model_dirs:
        experiment = model_dir.relative_to(args.output_root).parts[0]
        ply = model_dir / f"point_cloud/iteration_{args.iteration}/point_cloud.ply"
        kernel = args.kernel or read_kernel(model_dir)
        if kernel not in CONTROL_POINT or not ply.exists():
            print(f"[skip] {model_dir}: kernel={kernel}, ply exists={ply.exists()}")
            continue
        ctrl = CONTROL_POINT[kernel](load_inter_points(ply))   # (N, axes, 2)
        for axis, name in enumerate("uv"):
            path = out_root / experiment / f"{model_dir.name}_{name}.pdf"
            plot(ctrl[:, axis], path)
        print(f"[ok] {experiment}/{model_dir.name} ({kernel}, {len(ctrl)} Gaussians)")


if __name__ == "__main__":
    main()
