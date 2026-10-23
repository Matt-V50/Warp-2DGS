"""
Collect the metrics written by scripts/run_experiments.py into CSV tables.

    python scripts/collect_results.py --dataset mip360 --output_root output
    python scripts/collect_results.py --dataset tnt_mesh --output_root output --experiments "pwl*"

Novel view synthesis (mip360, tnt, blender_image) reads training_time.json and results_iter<N>.json
of every <output_root>/<experiment>/<dataset folder>/<scene>[_<factor>] and writes one row per run.
tnt_mesh reads the official evaluation (*.prf_tau_plotstr.txt) and also writes an
experiment x scene F-score table. A per-experiment mean is printed in both cases.
"""

import csv
import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

NVS_LAYOUT = {
    "mip360":        "Mip-NeRF360/360_v2",
    "tnt":           "tanks_and_temples",
    "blender_image": "BlenderImage",
}
TNT_MESH_LAYOUT = "TNT_GOF/TrainingSet"
METRICS = ["PSNR", "SSIM", "LPIPS"]
PRF_KEYS = ["precision", "recall", "fscore", "tau", "plot_stretch"]


def split_factor(name):
    """bonsai_4 -> (bonsai, 4); scenes without a numeric suffix are full resolution."""
    scene, _, factor = name.rpartition("_")
    return (scene, int(factor)) if scene and factor.isdigit() else (name, 1)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows -> {path}")


def print_means(rows, group_keys, value_keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in group_keys)].append(row)
    print("\n" + "\t".join(group_keys + value_keys + ["n"]))
    for key, members in sorted(groups.items()):
        means = []
        for v in value_keys:
            values = [m[v] for m in members if m[v] is not None]
            means.append(f"{sum(values) / len(values):.4f}" if values else "-")
        print("\t".join(map(str, key)) + "\t" + "\t".join(means) + f"\t{len(members)}")


def collect_nvs(args):
    pattern = f"{args.experiments}/{NVS_LAYOUT[args.dataset]}/*/training_time.json"
    rows = []
    for timing_file in sorted(args.output_root.glob(pattern)):
        run_dir = timing_file.parent
        experiment = run_dir.relative_to(args.output_root).parts[0]
        scene, factor = split_factor(run_dir.name)
        timing = json.loads(timing_file.read_text())
        row = {"experiment": experiment, "scene": scene, "factor": factor}
        for it in args.iterations:
            result_file = run_dir / f"results_iter{it}.json"
            if not result_file.exists():
                print(f"[missing] {result_file}")
                result = {}
            else:
                result = json.loads(result_file.read_text())
            for split, metric in ((s, m) for s in ("train", "test") for m in METRICS):
                row[f"{metric.lower()}_{it // 1000}k_{split}"] = result.get(split, {}).get(metric)
            row[f"gaussians_{it // 1000}k"] = result.get("count")
        row.update(train_time=timing["train_times"], train_render_time=timing["train_render_times"],
                   train_optimal_time=timing["train_optimal_times"])
        rows.append(row)
    if not rows:
        print("No runs matched:", args.output_root / pattern)
        return

    write_csv(args.out or args.output_root / f"results_{args.dataset}.csv", rows)
    last = args.iterations[-1] // 1000
    print_means(rows, ["experiment", "factor"],
                [f"{m.lower()}_{last}k_test" for m in METRICS] + [f"gaussians_{last}k"])


def collect_tnt_mesh(args):
    it = args.iterations[-1]
    pattern = f"{args.experiments}/{TNT_MESH_LAYOUT}/*/train/ours_{it}/evaluation/*.prf_tau_plotstr.txt"
    rows = []
    for prf_file in sorted(args.output_root.glob(pattern)):
        parts = prf_file.relative_to(args.output_root).parts
        values = [float(t) for t in prf_file.read_text().split()]
        if len(values) < 3:
            print(f"[skip] {prf_file}: only {len(values)} values")
            continue
        rows.append({"experiment": parts[0], "scene": parts[3], "iteration": it,
                     **dict(zip(PRF_KEYS, values))})
    if not rows:
        print("No runs matched:", args.output_root / pattern)
        return

    out = args.out or args.output_root / "results_tnt_mesh.csv"
    write_csv(out, rows)
    # experiment x scene F-score table, as reported in the paper
    scenes = sorted({r["scene"] for r in rows})
    fscore = {(r["experiment"], r["scene"]): r["fscore"] for r in rows}
    table = []
    for exp in sorted({r["experiment"] for r in rows}):
        values = {s: fscore.get((exp, s)) for s in scenes}
        got = [v for v in values.values() if v is not None]
        table.append({"experiment": exp, **values, "mean": sum(got) / len(got), "n": len(got)})
    write_csv(out.with_name(out.stem + "_fscore.csv"), table)
    print_means(rows, ["experiment"], ["precision", "recall", "fscore"])


def main():
    parser = ArgumentParser(description="Collect experiment metrics into CSV tables")
    parser.add_argument("--dataset", required=True, choices=list(NVS_LAYOUT) + ["tnt_mesh"])
    parser.add_argument("--output_root", required=True, type=Path, help="--output_root of run_experiments.py")
    parser.add_argument("--experiments", default="*", help="glob over experiment folders, e.g. 'pwl*'")
    parser.add_argument("--iterations", nargs="+", type=int, default=[7000, 30000])
    parser.add_argument("--out", type=Path, default=None, help="default: <output_root>/results_<dataset>.csv")
    args = parser.parse_args()
    if args.dataset == "tnt_mesh":
        collect_tnt_mesh(args)
    else:
        collect_nvs(args)


if __name__ == "__main__":
    main()
