#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
from typing import List, Optional

import torch
from scene import Scene
import os
import json
from tqdm import tqdm
from gaussian_renderer import render
from scene.cameras import Camera
from utils.general_utils import safe_state
from utils.loss_utils import ssim
from utils.image_utils import psnr
# from lpipsPyTorch import lpips
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPSTorch
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.render_utils import save_img_u8

lpips_model = LPIPSTorch(net_type="vgg", normalize=True).to(torch.device("cuda"))
lpips_model.eval()

def lpips(x, y):
    x = torch.clamp(x, 0.0, 1.0)
    y = torch.clamp(y, 0.0, 1.0)
    return lpips_model(x, y)

def evaluate_set(name, views: List[Camera], gaussians, pipeline, background, save_path:Optional[Path]=None):
    """
    Evaluate metrics for a set of views without saving images to disk.
    
    Args:
        name: Name of the set (train/test)
        views: List of camera views
        gaussians: Gaussian model
        pipeline: Pipeline parameters
        background: Background color tensor
    
    Returns:
        Dictionary containing mean metrics and per-view metrics
    """
    ssims = []
    psnrs = []
    lpipss = []
    per_view_metrics = {"SSIM": {}, "PSNR": {}, "LPIPS": {}}
    
    
    for idx, view in enumerate(tqdm(views, desc=f"Evaluating {name}")):
        # Render the image
        render_package = render(view, gaussians, pipeline, background)
        rendering = render_package["render"]  # RGB
        gt = view.original_image[0:3, :, :].cuda()
        
        # Add batch dimension for metric computation1
        rendering = rendering.unsqueeze(0)
        gt = gt.unsqueeze(0)
        
        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)
            # save the image
            save_img_u8(rendering[0].permute(1,2,0).cpu().numpy(), save_path / f'{view.image_name}.png')
        
        # Compute metrics
        ssim_val = ssim(rendering, gt)
        psnr_val = psnr(rendering, gt)
        lpips_val = lpips(rendering, gt)
        
        ssims.append(ssim_val)
        psnrs.append(psnr_val)
        lpipss.append(lpips_val)
        
        # Store per-view metrics
        # view_name = '{0:05d}.png'.format(idx)
        view_name = view.image_name if hasattr(view, 'image_name') else '{0:05d}'.format(idx)
        if not view_name.endswith('.png'):
            view_name = f'{view_name}.png'
        per_view_metrics["SSIM"][view_name] = ssim_val.item() if torch.is_tensor(ssim_val) else ssim_val
        per_view_metrics["PSNR"][view_name] = psnr_val.item() if torch.is_tensor(psnr_val) else psnr_val
        per_view_metrics["LPIPS"][view_name] = lpips_val.item() if torch.is_tensor(lpips_val) else lpips_val
    
    # Compute mean metrics
    mean_ssim = torch.tensor(ssims).mean().item() if ssims else 0
    mean_psnr = torch.tensor(psnrs).mean().item() if psnrs else 0
    mean_lpips = torch.tensor(lpipss).mean().item() if lpipss else 0
    
    mean_metrics = {
        "SSIM": mean_ssim,
        "PSNR": mean_psnr,
        "LPIPS": mean_lpips
    }
    
    return mean_metrics, per_view_metrics

def render_set(name, views: List[Camera], gaussians, pipeline, background, save_path:Optional[Path], factor=4, T_limit=None,
               save_depth=False):
    if T_limit is not None:
        pipeline.T_limit = T_limit
    
    if factor != 1:
        save_path = save_path.with_name(f"{save_path.name}_x{factor}")
    if T_limit is not None:
        save_path = save_path.with_name(f"{save_path.name}_T{T_limit:.1f}")
    
    depth_save_path = save_path.with_name(save_path.name + "_depth")
    for idx, view in enumerate(tqdm(views, desc=f"Evaluating {name}")):
        # Render the image
        image_path = save_path / f'{view.image_name}.png'
        if image_path.exists() and save_depth and depth_save_path.exists():
            continue
        old_image_height = view.image_height
        old_image_width = view.image_width
        # old_image_name = view.image_name
        view.image_height *= factor
        view.image_width *= factor
        view.image_name += f"_x{factor}"
        render_package = render(view, gaussians, pipeline, background)
        rendering = render_package["render"]  # RGB
        depth = render_package["depth"]  # Depth
        gt = view.original_image[0:3, :, :].cuda()
        
        # Add batch dimension for metric computation
        rendering = rendering.unsqueeze(0)
        depth = depth.unsqueeze(0)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        gt = gt.unsqueeze(0)
        
        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)
            # save the image
            save_img_u8(rendering[0].permute(1,2,0).cpu().numpy(), save_path / f'{view.image_name}.png')
            if save_depth:
                depth_save_path.mkdir(parents=True, exist_ok=True)
                depth_vis = depth[0].permute(1, 2, 0).expand(-1, -1, 3).cpu().numpy()
                save_img_u8(depth_vis, depth_save_path / f'{view.image_name}.png')
        view.image_height = old_image_height
        view.image_width = old_image_width
        return
        
        
def evaluate_all(dataset: ModelParams, iteration: int, pipeline: PipelineParams, 
                 skip_train: bool, skip_test: bool, save_results: bool = True, save_image: bool = False,
                 save_depth: bool = False):
    """
    Evaluate metrics for train and test sets.
    
    Args:
        dataset: Model parameters
        iteration: Iteration to load
        pipeline: Pipeline parameters
        skip_train: Whether to skip train set evaluation
        skip_test: Whether to skip test set evaluation
        save_results: Whether to save results to JSON files
    """
    with torch.no_grad():
        gaussians = GaussianModel.from_model_params(dataset)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        full_dict = {"count": scene.gaussians.get_xyz.shape[0]}
        per_view_dict = {}
        save_path = None
        if save_image:
            save_path = Path(dataset.model_path)
        
        # Evaluate train set
        if not skip_train:
            print("\n" + "="*50)
            print("Evaluating Train Set")
            print("="*50)
            
            save_path_train = save_path / "train" if save_path else None
            train_views = scene.getTrainCameras()
            train_mean, train_per_view = evaluate_set(
                "train", train_views, gaussians, pipeline, background, save_path_train
            )
            # render_set(
            #     "train", train_views, gaussians, pipeline, background, save_path_train, factor=4, T_limit=0.4
            # )
            # render_set(
            #     "train", train_views, gaussians, pipeline, background, save_path_train, factor=1, T_limit=0.5, save_depth=save_depth
            # )
            # render_set(
            #     "train", train_views, gaussians, pipeline, background, save_path_train, factor=1, T_limit=0, save_depth=save_depth
            # )
            # return
            
            full_dict["train"] = train_mean
            per_view_dict["train"] = train_per_view
            
            print(f"\nTrain Set Results (iteration {scene.loaded_iter}):")
            print(f"  SSIM : {train_mean['SSIM']:>12.7f}")
            print(f"  PSNR : {train_mean['PSNR']:>12.7f}")
            print(f"  LPIPS: {train_mean['LPIPS']:>12.7f}")
        
        # Evaluate test set
        if not skip_test:
            print("\n" + "="*50)
            print("Evaluating Test Set")
            print("="*50)
            
            test_views = scene.getTestCameras()
            save_path_test = save_path / "test" if save_path else None
            test_mean, test_per_view = evaluate_set(
                "test", test_views, gaussians, pipeline, background, save_path_test,
                # save_depth=save_depth
            )
            
            # render_set(
            #     "test", test_views, gaussians, pipeline, background, save_path_test, factor=1, T_limit=0.5, save_depth=save_depth
            # )
            # render_set(
            #     "test", test_views, gaussians, pipeline, background, save_path_test, factor=1, T_limit=0, save_depth=save_depth
            # )
        # return
            full_dict["test"] = test_mean
            per_view_dict["test"] = test_per_view
            
            print(f"\nTest Set Results (iteration {scene.loaded_iter}):")
            print(f"  SSIM : {test_mean['SSIM']:>12.7f}")
            print(f"  PSNR : {test_mean['PSNR']:>12.7f}")
            print(f"  LPIPS: {test_mean['LPIPS']:>12.7f}")
        
        # Save results to JSON
        if save_results:
            results_path = os.path.join(dataset.model_path, f"results_iter{scene.loaded_iter}.json")
            per_view_path = os.path.join(dataset.model_path, f"per_view_iter{scene.loaded_iter}.json")
            
            with open(results_path, 'w') as fp:
                json.dump(full_dict, fp, indent=2)
            with open(per_view_path, 'w') as fp:
                json.dump(per_view_dict, fp, indent=2)
            
            print(f"\nResults saved to:")
            print(f"  {results_path}")
            print(f"  {per_view_path}")
        
        return full_dict, per_view_dict


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    
    # Set up command line argument parser
    parser = ArgumentParser(description="Evaluation script - compute metrics without saving images")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iterations", nargs='+', default=[7000, 30000], type=int, 
                        help="List of iterations to evaluate (default: 7000 30000)")
    parser.add_argument("--skip_train", action="store_true", help="Skip train set evaluation")
    parser.add_argument("--skip_test", action="store_true", help="Skip test set evaluation")
    parser.add_argument("--no_save", action="store_true", help="Do not save results to JSON files")
    parser.add_argument("--save_image", action="store_true", help="Save rendered images to disk", default=False)
    parser.add_argument("--save_depth", action="store_true", help="Save depth maps to disk", default=False)
    parser.add_argument("--quiet", action="store_true", help="Suppress progress bars")
    args = get_combined_args(parser)
    
    print("Evaluating " + args.model_path)
    print(f"Iterations to evaluate: {args.iterations}")
    
    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    tm_lpips = LPIPSTorch(net_type='vgg', normalize=True).to(device)
    tm_lpips.eval()
    
    # Loop through all iterations
    for iteration in args.iterations:
        print("\n" + "="*70)
        print(f"EVALUATING ITERATION {iteration}")
        print("="*70)
        
        evaluate_all(
            model.extract(args), 
            iteration, 
            pipeline.extract(args), 
            args.skip_train, 
            args.skip_test,
            save_results=not args.no_save,
            save_image=args.save_image,
            save_depth=args.save_depth
        )