import json
from pathlib import Path

from itertools import product
import json
from pathlib import Path
from pandas import DataFrame as Df
import pandas as pd
from tqdm import tqdm

gs_output_root = Path("/mnt/cviss/Matt/GS-Output")
items = []
outputs = gs_output_root.glob("*_gradient_accumulate/Mip-NeRF360/*/*/training_time.json")
for output in tqdm(outputs):
    experiment = output.relative_to(gs_output_root).parts[0]
    # root = gs_output_root / method
    root = gs_output_root / experiment
    
    rel_path = output.relative_to(root)
    scene_rel = rel_path.parent
    dataset_name = scene_rel.parts[0]
    scene_name = Path(*scene_rel.parts[1:])
    try:
        scene_name_new, downsampling = str(scene_name).rsplit('_', 1)
        downsampling = str(int(downsampling))
        scene_name = scene_name_new
    except:
        downsampling = "1"

    dataset_dir = root / dataset_name
    scene_dir = root / scene_rel

    training_time_file = scene_dir / "training_time.json"
    
    item = {
        "dataset": dataset_name,
        "scene": scene_name,
        "method": experiment,
        "downsampling": downsampling,
    }
    with open(training_time_file, 'r') as f:
        training_time_data = json.load(f)
    item["_times"] = training_time_data["train_times"]
    item["_train_render_times"] = training_time_data["train_render_times"]  
    item["_train_optimal_times"] = training_time_data["train_optimal_times"]
    for step in [7000, 30000]:
        eval_file = scene_dir / f"results_iter{step}.json"
        with open(eval_file, 'r') as f:
            eval_data = json.load(f)
        step = step // 1000
        item.update({
            f"psnr_{step}k_train": eval_data["train"]["PSNR"],
            f"ssim_{step}k_train": eval_data["train"]["SSIM"],
            f"lpips_{step}k_train": eval_data["train"]["LPIPS"],
            f"psnr_{step}k_test": eval_data["test"]["PSNR"],
            f"ssim_{step}k_test": eval_data["test"]["SSIM"],    
            f"lpips_{step}k_test": eval_data["test"]["LPIPS"],
        })
        if step == 30:
            item["gs_number"] = eval_data["count"]
    items.append(item)

df = pd.DataFrame(items)
df["gpu"] = "RTX 4090"
df["resolution"] = None
df = df[["method", "dataset", "scene", "gpu", "downsampling", "psnr_7k_train", "ssim_7k_train", "lpips_7k_train", "psnr_7k_test", "ssim_7k_test", "lpips_7k_test", "psnr_30k_train", "ssim_30k_train", "lpips_30k_train", "psnr_30k_test", "ssim_30k_test", "lpips_30k_test", "resolution", "gs_number", "_times", "_train_render_times", "_train_optimal_times"]]
df.to_csv("results_summary.csv", index=False, sep='\t')
