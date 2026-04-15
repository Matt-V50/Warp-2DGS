from concurrent.futures import process
from dataclasses import dataclass
from itertools import product
import os
from pathlib import Path
import pty
import select
import subprocess
import sys
import struct
import fcntl
import termios

def run_with_live_output(cmd):
    """运行命令并实时显示输出，正确处理 tqdm"""
    def get_terminal_size():
        try:
            ws = fcntl.ioctl(sys.stdout, termios.TIOCGWINSZ, b'\x00' * 8)
            rows, cols = struct.unpack('HHHH', ws)[:2]
            if rows > 0 and cols > 0:
                return rows, cols
        except OSError:
            pass
        return 24, 80
    
    print(f"Running command: \n{cmd}")
    master_fd, slave_fd = pty.openpty()
    # 初始化 pty 终端大小（同步外层终端）
    rows, cols = get_terminal_size()
    winsize = struct.pack('HHHH', rows, cols, 0, 0)
    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)

    process = subprocess.Popen(
        cmd,
        shell=True,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
    )
    os.close(slave_fd)

    output_lines = []

    try:
        while True:
            ready, _, _ = select.select([master_fd], [], [], 0.1)
            if ready:
                try:
                    data = os.read(master_fd, 4096).decode('utf-8', errors='replace')
                    if data:
                        print(data, end='', flush=True)
                        output_lines.append(data)
                except OSError:
                    break

            if process.poll() is not None:
                while True:
                    try:
                        data = os.read(master_fd, 4096).decode('utf-8', errors='replace')
                        if not data:
                            break
                        print(data, end='', flush=True)
                        output_lines.append(data)
                    except OSError:
                        break
                break
    finally:
        os.close(master_fd)
    process.wait() 
    return process.returncode, ''.join(output_lines)


def get_train_cmd(input, output, experiment="GS_Experiment", image_dir=None, inter_points_lr=0.005, save_images=False):

    # 路径配置
    image_root = f"/home/matt/cviss/Matt/Dataset/{input}"  # Blendswap/Render/pick/13078_toad
    output_base_dir = f"/home/matt/cviss/Matt/GS-Output"
    output_full_dir = f"{output_base_dir}/{experiment}/{output}"  # Blendswap/Render/pick/13078_toad'strategy}/{output}"  # pick/13078_toad
    cmd = (
        f"OMP_NUM_THREADS=4 "
        f"CUDA_VISIBLE_DEVICES=0 "
        f"python train.py "
        f"--disable_viewer "
        f"-s {image_root} "
        f"-m {output_full_dir} "
        f"{f'-i {image_dir} ' if image_dir is not None else ''}"
        f"-r 1 "
        f"--eval "
        f"--iterations 30000 "
        f"--save_iterations 7000 30000 "
        f"--inter_points_lr {inter_points_lr}",  # 0.005 default
    )
    

    # ============================================================
    # 评估命令
    # ============================================================
    eval_cmd = (
        f"OMP_NUM_THREADS=4 "
        f"CUDA_VISIBLE_DEVICES=0 "
        f"python evaluate_metrics.py -s {image_root} -m {output_full_dir} --iterations 7000 30000 "
        f"{'--save_image ' if save_images else ''}"
    )
    # eval_cmd = None
    return cmd, eval_cmd

def get_compile_cmd(header: str, dof: int) -> tuple:
    template = f'''#define CONTROL_POINTS 2 // Default 1
#define RESCALE_FACTOR 3.0f // Default 3.0f
#define DOF {dof}  // Degrees of freedom: Pchip(2), SmoothSigmoid(1), Pwl3(2), Pwl4(4), Pwl4Accumulate(6), Pwl5Symmetric(2), Pwl_abs(1), Pwl5Asymmetric(4), Pwl4Phase(4)

#include "{header}"'''
    Path(
        "/home/matt/Projects/git/2d-gaussian-splatting/"
        "submodules/diff-surfel-rasterization/cuda_rasterizer/deformation.h"
    ).write_text(template)
    
    compile_cmd = ("pip install --no-build-isolation  ./submodules/diff-surfel-rasterization/", )
    
    update_train_cmd = (
        f"sed -i '119s/dof=[0-9]*/dof={dof}/' /home/matt/Projects/git/2d-gaussian-splatting/train.py", 
    )
    return compile_cmd, update_train_cmd

@dataclass
class Experiment:
    name: str
    header: str
    dof: int


if __name__ == "__main__":
    # for factor, strategy in product([2], ["mcmc", "default"]):
    #     # 示例用法
    #     cmd, eval_cmd = get_train_cmd(input="Rogers/Tower_0529", output=f"Rogers/Tower_0529_{factor}", factor=factor, strategy=strategy)

    #     run_with_live_output(cmd)
    #     run_with_live_output(eval_cmd)
    
    experiments = [
        Experiment("2DGS_dev_pchip_x2_3sigma", "pchip.h", 2),
        Experiment("2DGS_dev_pchip3_x2_3sigma_grad_fixed", "pchip3_fixed.h", 2),
        Experiment("2DGS_dev_pchip5_x2_symmetric", "pchip5_symmetric.h", 2),
        Experiment("2DGS_dev_pwl3_x2_3sigma_2grad_fixed_again", "pwl3.h", 2),
        Experiment("2DGS_dev_pwl3_y0_x2_3sigma", "pwl3_y0.h", 1),
        # Experiment("2DGS_dev_pwl_abs_x2_3sigma", "pwl_abs.h", 1),
        # Experiment("2DGS_dev_pwl4_x2_3sigma", "pwl4.h", 4),
        # Experiment("2DGS_dev_pwl4_segment_x2_3sigma_bug_fixed", "pwl4_segment.h", 4),
        Experiment("2DGS_dev_pwl5_asymmetric_x2_3sigma", "pwl5_asymmetric.h", 4),
        Experiment("2DGS_dev_pwl5_asymmetric_y0_x2_3sigma", "pwl5_asymmetric_y0.h", 4),
        Experiment("2DGS_dev_pwl5_symmetric_x2_3sigma", "pwl5_symmetric.h", 2),
        Experiment("2DGS_dev_empty", "empty.h", 1),
        # Experiment("2DGS_dev_pwl4_accumulate_3sigma", "pwl4_accumulate.h", 6),  # uses too much shared data
    ]
    
    # MipNerf-360 dataset
    dataset_root = Path("/mnt/cviss/Matt/Dataset")
    mip_root = dataset_root / "Mip-NeRF360/360_v2/"
    blender_image_root = dataset_root / "BlenderImage"

    scenes = [("Rogers/Tower_0529", "Rogers/Tower_0529")]
    scenes = []
    scenes.extend([
        (f"Mip-NeRF360/360_v2/{d.name}", f"Mip-NeRF360/360_v2/{d.name}")
        for d in mip_root.iterdir() if d.is_dir()
    ])
    # scenes = []
    # scenes.extend([
    #     (f"BlenderImage/{d.name}", f"BlenderImage/{d.name}")
    #     for d in blender_image_root.iterdir() if d.is_dir()
    # ])
    factors = [4, 8]
    inter_points_lrs = [0.005, 0.0025]
    save_images = True
    
    parameters = product(experiments, inter_points_lrs, scenes, factors)
    total_runs = len(experiments) * len(inter_points_lrs) * len(scenes) * len(factors)
    old_experiment = None
    for i, (exp, inter_points_lr, scene, factor) in enumerate(parameters):
        print(f"Running experiment {i+1}/{total_runs}: {exp.name}, inter_points_lr={inter_points_lr}, scene={scene[0]}, factor={factor}")
        experiment_name = exp.name
        if inter_points_lr != 0.005:
            experiment_name += f"_inter_lr_{inter_points_lr}"
        experiment_name += f"_gradient_accumulate"
        input_path, output_base = scene
        output_path = f"{output_base}_{factor}"
        # scene_dir = f"{input_path}/dataset"  # for BlenderImage dataset
        scene_dir = f"{input_path}"  # for Non BlenderImage dataset
        # scene_dir += f"_{factor}" if factor != 1 else "" # for BlenderImage dataset
        
        trained_flag = os.path.exists(f"/mnt/cviss/Matt/GS-Output/{experiment_name}/{output_base}_{factor}/training_time.json")
        eval_flag = os.path.exists(f"/mnt/cviss/Matt/GS-Output/{experiment_name}/{output_base}_{factor}/results_iter30000.json")
        if trained_flag and eval_flag:
            continue
        if old_experiment != experiment_name:
            compile_cmd, update_train_cmd = get_compile_cmd(exp.header, exp.dof)
            for cmd in [compile_cmd, update_train_cmd]:
                return_code, _ = run_with_live_output(cmd)
                if return_code != 0:
                    print(f"Command failed with return code {return_code}: {cmd}")
                    sys.exit(1)
            old_experiment = experiment_name
        
        cmd, eval_cmd = get_train_cmd(
            input=scene_dir,
            output=output_path,
            experiment=experiment_name,
            image_dir=f"images_{factor}",  # comment out if using BlenderImage dataset
            inter_points_lr=inter_points_lr,
            # save_images=save_images,  # comment out if NOT using BlenderImage dataset
        )
        if not trained_flag:
            run_with_live_output(cmd)
        if not eval_flag:
            run_with_live_output(eval_cmd)