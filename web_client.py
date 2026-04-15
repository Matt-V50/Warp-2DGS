# add /home/matt/Projects/git/Gaussian-GUI/client_dependencies to sys.path before running this script

import sys
import os

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from utils.general_utils import safe_state
sys.path.append("/home/matt/Projects/git/Gaussian-GUI/client_dependencies")
sys.path.append("/home/matt/Projects/git/Gaussian-GUI/export")

from argparse import ArgumentParser

from gaussian_renderer import render
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
import asyncio
from typing import Dict
from client_utils import RenderStreamTrack, OrbitCamera, CONFIG, setup
class GS2DTrack(RenderStreamTrack):
    """
    A video track that returns an animated flag.
    """

    def __init__(self, oribit_camera: OrbitCamera, opt=None, gaussian_model=None):
        super().__init__(oribit_camera, opt, gaussian_model)  # don't forget this!
        pass
        
    
    def render(self, camera: Camera, *args, **kwargs) -> Dict:
        return render(camera, self.gs_model, self.opt, self.bg_color, *args, **kwargs)
    
    
async def main():
    name = "2D-GS"
    parser = ArgumentParser(description="GUI option")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')
    
    
    parser.add_argument('--server_ip', type=str, default="127.0.0.1")
    parser.add_argument('--server_port', type=int, default=8080)
    args = get_combined_args(parser)
    
    args.source_path = args.source_path.replace("/content", "/home/matt/colab_data/content")
    print("Rendering " + args.model_path)

    # parser.add_argument('-m', '--model_path', type=str, default="./output/figurines")
    safe_state(args.quiet)
    # python gui.py --model_path ./output/be6c999c-7 --iteration 30000

    dataset = model.extract(args)
    iteration = args.iteration
    pipeline = pipeline.extract(args)
    
    opt = CONFIG()
    opt.MODEL_PATH = args.model_path
    opt.SCENE_GAUSSIAN_ITERATION = args.iteration
    opt.SCENE_PCD_PATH = os.path.join(opt.MODEL_PATH, f'point_cloud/iteration_{str(opt.SCENE_GAUSSIAN_ITERATION)}/point_cloud.ply')
    opt.MLP_PATH = os.path.join(opt.MODEL_PATH, f'point_cloud/iteration_{str(opt.SCENE_GAUSSIAN_ITERATION)}')
    opt.depth_ratio = pipeline.depth_ratio
    
    gs_model = GaussianModel(dataset.sh_degree)
    gs_model.load_ply(opt.SCENE_PCD_PATH)
    
    scene = Scene(dataset, gs_model, load_iteration=iteration, shuffle=False)
    train_cameras = scene.getTrainCameras()
    default_camera = train_cameras[0]
    orbit_camera = OrbitCamera(W=default_camera.image_width, H=240*4, r=2, fovy=60)
    orbit_camera.set_from_train_camera(default_camera)
    stream_track = GS2DTrack(orbit_camera, gaussian_model=gs_model, opt=opt)
    pc = await setup(name, stream_track, args.server_ip, args.server_port, orbit_camera)
    await asyncio.sleep(300)
    await pc.close()

if __name__ == "__main__":
    
    asyncio.run(main())