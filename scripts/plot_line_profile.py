"""
Inverted gray value along a scanline, compared with ground truth (the "Plot" column of Figure 1).

    python scripts/plot_line_profile.py 2dgs.png warp2dgs.png gt.png
    python scripts/plot_line_profile.py 2dgs.png warp2dgs.png gt.png --points 120 340 260 340

Without --points, the first image opens and the two clicked points define the scanline; they are
printed so the same line can be reused with --points. Each rendering gets one plot,
<out_dir>/<image name>_profile.png, with the ground truth as a dashed line.
"""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def get_line_pixels(img_array, p0, p1, num_samples=None):
    length = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
    if num_samples is None:
        num_samples = int(np.ceil(length))

    xs = np.linspace(p0[0], p1[0], num_samples)
    ys = np.linspace(p0[1], p1[1], num_samples)

    xi = np.clip(np.round(xs).astype(int), 0, img_array.shape[1] - 1)
    yi = np.clip(np.round(ys).astype(int), 0, img_array.shape[0] - 1)

    return img_array[yi, xi]  # (N, C)


def plot_profile(values, values_gt, save_path):
    gray = 1.0 - np.mean(values, axis=1) / 255.0  # grayscale + invert
    gray_gt = 1.0 - np.mean(values_gt, axis=1) / 255.0

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.plot(np.arange(len(gray)), gray, color='#7B2D8E', linewidth=4.5)
    ax.plot(np.arange(len(gray_gt)), gray_gt, color='gray', linewidth=2, linestyle='--')
    ax.set_xlim(0, len(gray) - 1)
    ax.set_ylim(0, 1)
    # 10 x 10 grid without tick labels
    ax.set_xticks(np.linspace(0, len(gray) - 1, 11))
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.yaxis.set_tick_params(length=0, labelsize=0)
    ax.xaxis.set_tick_params(length=0, labelsize=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, which='both', axis='y', linestyle='--', color='gray', alpha=0.7)
    ax.grid(True, which='both', axis='x', linestyle='--', color='gray', alpha=0.7)
    fig.tight_layout(pad=0)
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)
    print(f"Saved: {save_path}")


def pick_points(img):
    pts = []
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)
    ax.set_title("Click 2 points, then close the window")

    def onclick(event):
        if event.inaxes != ax or len(pts) >= 2:
            return
        if fig.canvas.toolbar.mode != '':
            return
        pts.append((event.xdata, event.ydata))
        ax.plot(event.xdata, event.ydata, 'x', color='magenta', markersize=10, markeredgewidth=2)
        if len(pts) == 2:
            ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]],
                    '--', color='magenta', linewidth=1)
            ax.set_title(f"Line: ({pts[0][0]:.0f},{pts[0][1]:.0f}) -> ({pts[1][0]:.0f},{pts[1][1]:.0f})  -- close window")
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return pts


def main():
    parser = ArgumentParser(description="Plot the inverted gray value along a scanline")
    parser.add_argument("images", nargs="+", type=Path, help="renderings to compare")
    parser.add_argument("gt", type=Path, help="ground truth image")
    parser.add_argument("--points", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"),
                        help="scanline end points in pixels; omit to pick them interactively")
    parser.add_argument("--out_dir", type=Path, default=None, help="default: folder of the first image")
    args = parser.parse_args()

    images = [np.array(Image.open(p).convert('RGB')) for p in args.images]
    img_gt = np.array(Image.open(args.gt).convert('RGB'))
    assert all(img.shape == img_gt.shape for img in images), "Images must be the same size"

    if args.points:
        p0, p1 = tuple(args.points[:2]), tuple(args.points[2:])
    else:
        pts = pick_points(images[0])
        if len(pts) < 2:
            raise SystemExit("Need 2 points.")
        p0, p1 = pts
    print(f"Points: --points {p0[0]:.1f} {p0[1]:.1f} {p1[0]:.1f} {p1[1]:.1f}")

    out_dir = args.out_dir or args.images[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)
    vals_gt = get_line_pixels(img_gt, p0, p1)
    for path, img in zip(args.images, images):
        plot_profile(get_line_pixels(img, p0, p1), vals_gt, out_dir / f"{path.stem}_profile.png")


if __name__ == "__main__":
    main()
