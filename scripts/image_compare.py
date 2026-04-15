import sys
import subprocess


def get_cmd(method):
    gt_file = f"/mnt/cviss/Matt/Dataset/BlenderImage/Test/dataset_4/train/0001.png"
    method_file = f"/mnt/cviss/Matt/GS-Output/{method}/BlenderImage/Test_4/train/0001.png"
    output_file = f"/home/matt/Projects/git/2d-gaussian-splatting/scripts/compare/{method}_comparison.png"
    return f"convert {gt_file} {method_file} -compose difference -composite -evaluate multiply 5 {output_file}"

def run(cmd):
    
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    output_lines = []
    try:
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                output_lines.append(output.strip())
    except Exception as e:
        print(f"Error: {e}")
    return output_lines

methods = [
    "2DGS_dev_empty_inter_lr_0.0025",
    "2DGS_dev_pchip3_x2_3sigma_grad_fixed_inter_lr_0.0025",
    "2DGS_dev_pwl3_x2_3sigma_2grad_fixed_again_inter_lr_0.0025",
]
for method in methods:
    cmd = get_cmd(method)
    print(f"Running command for {method}: {cmd}")
    run(cmd)