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

    return process.returncode, ''.join(output_lines)


def get_train_cmd(input, output, experiment="GS_Experiment", image_dir=None, inter_points_lr=0.005):

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
        f"python evaluate_metrics.py -s {image_root} -m {output_full_dir} --iterations 7000 30000"
    )
    # eval_cmd = None
    return cmd, eval_cmd


if __name__ == "__main__":
    # for factor, strategy in product([2], ["mcmc", "default"]):
    #     # 示例用法
    #     cmd, eval_cmd = get_train_cmd(input="Rogers/Tower_0529", output=f"Rogers/Tower_0529_{factor}", factor=factor, strategy=strategy)

    #     run_with_live_output(cmd)
    #     run_with_live_output(eval_cmd)
    
    
    # MipNerf-360 dataset
    dataset_root = Path("/mnt/cviss/Matt/Dataset")
    mip_root = dataset_root / "Mip-NeRF360/360_v2/"

    scenes = [("Rogers/Tower_0529", "Rogers/Tower_0529")]
    scenes = []
    scenes.extend([
        (f"Mip-NeRF360/360_v2/{d.name}", f"Mip-NeRF360/360_v2/{d.name}")
        for d in mip_root.iterdir() if d.is_dir()
    ])
    # scenes.remove(("Mip-NeRF360/360_v2/bicycle", "Mip-NeRF360/360_v2/bicycle"))
    factors = [4, 8]
    inter_points_lr = 0.0025
    experiment_name = "2DGS_dev_empty_fastgs_abs"
    if inter_points_lr != 0.005:
        experiment_name += f"_inter_lr_{inter_points_lr}"
    
    # 2DGS_dev_pchip_symmetric_33x2_3sigma_so_rhoo
    
    for input_path, output_base in scenes:
        for factor in factors:
            output_path = f"{output_base}_{factor}"
            cmd, eval_cmd = get_train_cmd(
                input=input_path,
                output=output_path,
                experiment=experiment_name,
                image_dir=f"images_{factor}",
                inter_points_lr=inter_points_lr
            )
            if not os.path.exists(f"/mnt/cviss/Matt/GS-Output/{experiment_name}/{output_base}_{factor}/training_time.json"):
                run_with_live_output(cmd)
            run_with_live_output(eval_cmd)