"""
Batch runner for Warp-2DGS experiments: kernels x scenes x resolutions, skipping finished stages.

Examples:
    # Mip-NeRF 360 at x4 and x8, the four kernels of the main paper
    python scripts/run_experiments.py --dataset mip360 --data_root /data --output_root output \\
        --kernels pwl pwl_sr pchip pchip_sr

    # Tanks and Temples mesh reconstruction + official F-score, preview the commands only
    python scripts/run_experiments.py --dataset tnt_mesh --data_root /data --output_root output --dry_run

Expected layout under --data_root:
    mip360         Mip-NeRF360/360_v2/<scene>/images_<factor>
    tnt            tanks_and_temples/<scene>/images
    blender_image  BlenderImage/<scene>/dataset[_<factor>]
    tnt_mesh       TNT_GOF/TrainingSet/<scene>  and  TanksAndTemples/trainingdata/<scene> (official GT)
"""

import os
import subprocess
from argparse import ArgumentParser
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Experiment:
    kernel: str = "pwl_sr"
    rescale: float = 3.0
    cutoff: float = -1.0                  # <= 0: rescale * sqrt(2)
    inter_points_lr: float = 0.0025
    densify_until_iter: Optional[int] = None
    name: Optional[str] = None            # output folder; None derives it from the settings

    @property
    def folder(self):
        if self.name is None:
            folder = self.kernel
            if self.rescale != 3.0:
                folder += f"_r{self.rescale:g}"
            if self.cutoff > 0:
                folder += f"_cutoff{self.cutoff:g}"
            if self.inter_points_lr != 0.0025:
                folder += f"_lr{self.inter_points_lr:g}"
            if self.densify_until_iter is not None:
                folder += f"_densify{self.densify_until_iter}"
            return folder
        # suffix rule of the original experiment folders, keeps them resumable
        folder = self.name
        if self.inter_points_lr != 0.005:
            folder += f"_inter_lr_{self.inter_points_lr}"
        if self.rescale != 3.0:
            folder += f"_rescale_{self.rescale:.1f}"
        if self.densify_until_iter is not None and self.densify_until_iter != 15_000:
            folder += f"_densify_until_{self.densify_until_iter}"
        return folder

    def train_args(self):
        args = (f"--kernel {self.kernel} --rescale {self.rescale} --cutoff {self.cutoff} "
                f"--inter_points_lr {self.inter_points_lr}")
        if self.densify_until_iter is not None:
            args += f" --densify_until_iter {self.densify_until_iter}"
        return args


@dataclass
class Run:
    """One (dataset, scene, resolution) target: where to read, where to write, extra flags."""
    scene: str
    source: Path
    output: str                           # relative to <output_root>/<experiment folder>
    train_args: str = ""
    render_args: str = ""
    gt: Optional[Path] = None             # tnt_mesh only


# ----------------------------------------------------------------------------
# Dataset presets
# ----------------------------------------------------------------------------

TNT_MESH_CONFIG = {
    # lambda_dist for training, TSDF fusion settings for render.py
    **{s: dict(lambda_dist=100, voxel_size=0.004, sdf_trunc=0.016, depth_trunc=3.0)
       for s in ["Barn", "Caterpillar", "Ignatius", "Truck"]},
    **{s: dict(lambda_dist=10, voxel_size=0.006, sdf_trunc=0.024, depth_trunc=4.5)
       for s in ["Meetingroom", "Courthouse"]},
}


def list_scenes(root: Path):
    return sorted(d.name for d in root.iterdir() if d.is_dir())


def mip360_runs(data_root, scenes, factors):
    root = data_root / "Mip-NeRF360/360_v2"
    for scene, f in product(scenes or list_scenes(root), factors):
        yield Run(scene, root / scene, f"Mip-NeRF360/360_v2/{scene}_{f}", f"-i images_{f} -r 1")


def tnt_runs(data_root, scenes, factors):
    root = data_root / "tanks_and_temples"
    for scene, f in product(scenes or list_scenes(root), factors):
        images, output = ("images", scene) if f == 1 else (f"images_{f}", f"{scene}_{f}")
        yield Run(scene, root / scene, f"tanks_and_temples/{output}", f"-i {images} -r 1")


def blender_image_runs(data_root, scenes, factors):
    root = data_root / "BlenderImage"
    for scene, f in product(scenes or list_scenes(root), factors):
        dataset = "dataset" if f == 1 else f"dataset_{f}"
        yield Run(scene, root / scene / dataset, f"BlenderImage/{scene}_{f}", "-r 1 --image_fitting")


def tnt_mesh_runs(data_root, scenes, factors):
    root = data_root / "TNT_GOF/TrainingSet"
    for scene in scenes or list(TNT_MESH_CONFIG):
        cfg = TNT_MESH_CONFIG[scene]
        yield Run(scene, root / scene, f"TNT_GOF/TrainingSet/{scene}",
                  train_args=f"-r 2 --depth_ratio 1.0 --lambda_dist {cfg['lambda_dist']} --test_iterations -1",
                  render_args=(f"--depth_ratio 1.0 --num_cluster 1 --voxel_size {cfg['voxel_size']} "
                               f"--sdf_trunc {cfg['sdf_trunc']} --depth_trunc {cfg['depth_trunc']}"),
                  gt=data_root / "TanksAndTemples/trainingdata" / scene)


DATASETS = {
    "mip360":        dict(runs=mip360_runs,        factors=[4, 8], stages=["train", "eval"]),
    "tnt":           dict(runs=tnt_runs,           factors=[1],    stages=["train", "eval"]),
    "blender_image": dict(runs=blender_image_runs, factors=[1],    stages=["train", "eval"]),
    "tnt_mesh":      dict(runs=tnt_mesh_runs,      factors=[1],    stages=["train", "render", "metrics"]),
}


# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------

def stage_command(stage, exp: Experiment, run: Run, model: Path, args):
    """Return (command, marker): the stage is skipped when marker exists."""
    it = args.iterations
    if stage == "train":
        cmd = (f"python train.py --disable_viewer -s {run.source} -m {model} --eval "
               f"--iterations {it} --save_iterations 7000 {it} {run.train_args} {exp.train_args()}")
        return cmd, model / "training_time.json"
    if stage == "eval":
        cmd = (f"python evaluate_metrics.py -s {run.source} -m {model} --iterations 7000 {it}"
               f"{' --save_image' if args.save_images else ''}")
        return cmd, model / f"results_iter{it}.json"
    if stage == "render":
        cmd = f"python render.py --iteration {it} -s {run.source} -m {model} --quiet {run.render_args}"
        return cmd, model / f"train/ours_{it}/fuse_post.ply"
    if stage == "metrics":
        # the official TNT evaluation needs its own environment (older open3d), see scripts/eval_tnt
        cmd = (f"conda run -n {args.metrics_env} --no-capture-output "
               f"python scripts/eval_tnt/run.py --dataset-dir {run.gt} "
               f"--traj-path {run.source}/{run.scene}_COLMAP_SfM.log "
               f"--ply-path {model}/train/ours_{it}/fuse_post.ply")
        return cmd, model / f"train/ours_{it}/evaluation/{run.scene}.prf_tau_plotstr.txt"
    raise ValueError(stage)


def run_command(cmd, gpu):
    """Run in the current terminal so tqdm progress bars render normally; returns the exit code."""
    env = dict(os.environ, OMP_NUM_THREADS="4", CUDA_VISIBLE_DEVICES=str(gpu))
    return subprocess.run(cmd, shell=True, cwd=REPO_ROOT, env=env).returncode


def main():
    parser = ArgumentParser(description="Batch runner for Warp-2DGS experiments")
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument("--kernels", nargs="+", default=["pwl_sr"])
    parser.add_argument("--rescales", nargs="+", type=float, default=[3.0])
    parser.add_argument("--cutoff", type=float, default=-1.0)
    parser.add_argument("--inter_points_lrs", nargs="+", type=float, default=[0.0025])
    parser.add_argument("--densify_until_iter", type=int, default=None)
    parser.add_argument("--name", default=None,
                        help="reuse an existing output folder (single kernel/rescale/lr only), "
                             "suffixes follow the original naming rule")
    parser.add_argument("--scenes", nargs="+", default=None, help="default: all scenes of the dataset")
    parser.add_argument("--factors", nargs="+", type=int, default=None, help="image downsampling factors")
    parser.add_argument("--stages", nargs="+", default=None, choices=["train", "eval", "render", "metrics"])
    parser.add_argument("--iterations", type=int, default=30_000)
    parser.add_argument("--save_images", action="store_true", help="eval: also write rendered images")
    parser.add_argument("--metrics_env", default="2d-gs-eval", help="tnt_mesh: conda env of the official evaluation")
    parser.add_argument("--gpu", default=0)
    parser.add_argument("--force", action="store_true", help="rerun stages even if their outputs exist")
    parser.add_argument("--dry_run", action="store_true", help="print the commands without running them")
    args = parser.parse_args()

    preset = DATASETS[args.dataset]
    stages = args.stages or preset["stages"]
    experiments = [Experiment(kernel=k, rescale=r, cutoff=args.cutoff, inter_points_lr=lr,
                              densify_until_iter=args.densify_until_iter, name=args.name)
                   for k, r, lr in product(args.kernels, args.rescales, args.inter_points_lrs)]
    if args.name and len(experiments) > 1:
        parser.error("--name needs a single kernel / rescale / inter_points_lr")
    runs = list(preset["runs"](args.data_root, args.scenes, args.factors or preset["factors"]))

    jobs = list(product(experiments, runs))
    for i, (exp, run) in enumerate(jobs, 1):
        model = args.output_root / exp.folder / run.output
        header = f"[{i}/{len(jobs)}] {exp.folder} | {run.output}"
        print(f"\n{'=' * len(header)}\n{header}\n{'=' * len(header)}")
        for stage in stages:
            cmd, marker = stage_command(stage, exp, run, model, args)
            if marker.exists() and not args.force:
                print(f"skip {stage}: {marker} exists")
                continue
            print(f"{stage}: {cmd}")
            if args.dry_run:
                continue
            code = run_command(cmd, args.gpu)
            if code != 0:
                print(f"{stage} failed with exit code {code}, skipping the remaining stages of this run")
                break


if __name__ == "__main__":
    main()
