"""
Simplified 3D Gaussian Splatting Viewer
- RGB rendering
- Camera controls (orbit, pan, zoom)
- Camera switching from training views
"""

import time
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import cv2
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R

from scene import Scene, GaussianModel
from scene.dataset_readers import CameraInfo
from gaussian_renderer import render
from utils.graphics_utils import focal2fov, fov2focal, getProjectionMatrix, getWorld2View2
from argparse import ArgumentParser
from arguments import ModelParams, get_combined_args

class GUICamera:
    def __init__(self, R, T, FoVx, FoVy, uid, 
                 image_height, image_width,
                 cx = None, cy = None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0,
                 data_device = "cuda"
                 ):
        
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.uid = uid
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy

        self.image_width = image_width
        self.image_height = image_height

        # a dirty hack to make sure that the feature width is always 200 # TODO: fix this
        # It doesn't make sense to have a fixed feature width, it should be a parameter
        self.feature_width = 100
        self.feature_height = int(self.feature_width * image_height / image_width)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

@dataclass
class RenderConfig:
    """Rendering configuration"""
    width: int = 640
    height: int = 480
    window_width: int = 640
    window_height: int = 480
    
    # Camera
    radius: float = 2.0
    
    # Gaussian model
    sh_degree: int = 3
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    white_background: bool = False
    
    # Paths (set after parsing args)
    model_path: str = ""
    load_iteration: int = 30000
    scene_pcd_path: str = ""
    
    debug: bool = False
    T_limit: float = -1.0
    depth_ratio: float = 0.0


class OrbitCamera:
    """Orbit camera for interactive viewing"""
    
    def __init__(self, width: int, height: int, radius: float = 2.0, fovy: float = 60.0):
        self.W = width
        self.H = height
        self.radius = radius
        self.fovy = fovy
        
        self.center = np.array([0, 0, 0], dtype=np.float32)
        self.rot = R.from_quat([0, 0, 0, 1])
        self.up = np.array([0, 1, 0], dtype=np.float32)
        self.right = np.array([1, 0, 0], dtype=np.float32)

    def set_from_train_camera(self, train_camera: CameraInfo):
        """Initialize from a training camera"""
        self.radius = 0
        self.rot = R.from_matrix(train_camera.R)
        self.center = train_camera.R @ train_camera.T
        self.fovy = math.degrees(train_camera.FoVy)
        self.W = train_camera.image_width
        self.H = train_camera.image_height

    @property
    def pose(self) -> np.ndarray:
        """Get camera pose matrix [R | t]"""
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        res[:3, 3] -= self.center
        
        # Convention transform for gaussian-splatting
        res[:3, 3] = -rot[:3, :3].T @ res[:3, 3]
        return res

    @property
    def intrinsics(self) -> np.ndarray:
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2])

    def orbit(self, dx: float, dy: float):
        """Rotate camera around center"""
        up = self.rot.as_matrix()[:3, 1]
        side = self.rot.as_matrix()[:3, 0]
        
        rotvec_x = up * np.radians(0.01 * dx)
        rotvec_y = side * np.radians(0.01 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta: float):
        """Zoom in/out"""
        self.radius -= 0.1 * delta

    def pan(self, dx: float, dy: float):
        """Pan camera"""
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, -dy, 0])


class GaussianRenderGUI:
    """Simplified GUI for 3D Gaussian Splatting rendering"""
    
    def __init__(
        self,
        config: RenderConfig,
        gaussian_model: GaussianModel,
        data_cameras: Optional[List[CameraInfo]] = None
    ):
        self.config = config
        self.gaussian_model = gaussian_model
        self.data_cameras = data_cameras or []
        self.camera_mapping = {cam.image_name: i for i, cam in enumerate(self.data_cameras)}
        self.camera_index = 0
        
        # Initialize camera
        self.camera = OrbitCamera(config.width, config.height, radius=config.radius)
        if self.data_cameras:
            self.camera.set_from_train_camera(self.data_cameras[0])
        
        # Background color
        bg = [1, 1, 1] if config.white_background else [0, 0, 0]
        self.bg_color = torch.tensor(bg, dtype=torch.float32, device="cuda")
        
        # Render buffer
        self.render_buffer = np.zeros((config.width, config.height, 3), dtype=np.float32)
        
        # Render modes
        self.render_rgb = True
        
        # Mouse state
        self.mouse_dragging = False
        self.mouse_panning = False
        self.mouse_pos = (0, 0)
        
        # FPS tracking
        self.fps_tracker = FPSTracker()
        
        # Load model
        print(f"Loading model from {config.scene_pcd_path}...")
        self.gaussian_model.load_ply(config.scene_pcd_path)
        print("Model loaded.")
        
        # Setup GUI
        dpg.create_context()
        self._setup_gui()

    def _setup_gui(self):
        """Setup DearPyGui interface"""
        # Texture for rendering
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.config.width, self.config.height,
                self.render_buffer,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture"
            )

        # Main window
        with dpg.window(tag="_primary_window", width=self.config.window_width + 250, height=self.config.window_height):
            dpg.add_image("_texture")
        dpg.set_primary_window("_primary_window", True)

        # Control panel
        with dpg.window(label="Controls", tag="_control_window", width=240, height=self.config.window_height, 
                       pos=[self.config.window_width + 10, 0]):
            
            dpg.add_text("FPS: 0", tag="_fps")
            dpg.add_separator()
            
            # Render options
            dpg.add_text("Render Mode:")
            dpg.add_radio_button(
                items=["RGB", "Surf-Depth", "Dist"],
                default_value="RGB",
                horizontal=True,
                callback=self._toggle_render,
            )
            dpg.add_slider_float(label="T_limit", default_value=-1.0, min_value=0.0, max_value=1.0, tag="_TLimit")
            
            dpg.add_separator()
            
            # Camera selection
            if self.data_cameras:
                dpg.add_text("Camera:")
                dpg.add_listbox(
                    items=[c.image_name for c in self.data_cameras],
                    num_items=8,
                    tag="_camera_list",
                    callback=self._on_camera_change
                )
            
            dpg.add_separator()
            
            # Info
            dpg.add_text("Current view:", tag="_info")
            
            dpg.add_separator()
            
            # Actions
            dpg.add_button(label="Save Screenshot", callback=self._save_screenshot)
            dpg.add_button(label="Reset Camera", callback=self._reset_camera)

        # Input handlers
        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=self._on_mouse_wheel)
            dpg.add_mouse_click_handler(dpg.mvMouseButton_Left, callback=self._on_left_click)
            dpg.add_mouse_release_handler(dpg.mvMouseButton_Left, callback=self._on_left_release)
            dpg.add_mouse_click_handler(dpg.mvMouseButton_Middle, callback=self._on_middle_click)
            dpg.add_mouse_release_handler(dpg.mvMouseButton_Middle, callback=self._on_middle_release)
            dpg.add_mouse_move_handler(callback=self._on_mouse_move)

        # Viewport
        dpg.create_viewport(
            title="3DGS Viewer",
            width=self.config.window_width + 260,
            height=self.config.window_height,
            resizable=False
        )

        # Theme (remove padding)
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme("_primary_window", theme)

        dpg.setup_dearpygui()
        dpg.show_viewport()

    # === Callbacks ===
    
    def _toggle_render(self, sender, value):
        self.render_rgb = value

    def _on_camera_change(self, sender, value):
        if value in self.camera_mapping:
            self.camera_index = self.camera_mapping[value]
            cam = self.data_cameras[self.camera_index]
            self.camera.set_from_train_camera(cam)
            dpg.set_value("_info", f"View: {cam.image_name}")

    def _on_mouse_wheel(self, sender, delta):
        if dpg.is_item_focused("_primary_window"):
            self.camera.scale(delta)

    def _on_left_click(self):
        self.mouse_dragging = True

    def _on_left_release(self):
        self.mouse_dragging = False

    def _on_middle_click(self):
        self.mouse_panning = True

    def _on_middle_release(self):
        self.mouse_panning = False

    def _on_mouse_move(self, sender, pos, user_data):
        if dpg.is_item_focused("_primary_window"):
            dx = self.mouse_pos[0] - pos[0]
            dy = self.mouse_pos[1] - pos[1]
            
            if self.mouse_dragging and (dx != 0 or dy != 0):
                self.camera.orbit(-dx * 30, dy * 30)
            
            if self.mouse_panning and (dx != 0 or dy != 0):
                self.camera.pan(-dx * 20, dy * 20)
        
        self.mouse_pos = pos

    def _save_screenshot(self):
        """Save current render to file"""
        timestamp = int(time.time())
        Path("./screenshots").mkdir(exist_ok=True)
        
        img = self.render_buffer.reshape(self.config.height, self.config.width, 3)
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        path = f"./screenshots/{timestamp}.png"
        cv2.imwrite(path, img)
        print(f"Screenshot saved: {path}")

    def _reset_camera(self):
        """Reset to first training camera"""
        if self.data_cameras:
            self.camera_index = 0
            self.camera.set_from_train_camera(self.data_cameras[0])

    # === Rendering ===
    
    def _build_camera(self) -> GUICamera:
        """Construct camera for rendering"""
        pose = self.camera.pose
        R = pose[:3, :3]
        t = pose[:3, 3]
        
        fovy = self.camera.fovy * (math.pi / 180.0)
        fy = fov2focal(fovy, self.config.height)
        fovx = focal2fov(fy, self.config.width)
        
        cam = GUICamera(
            image_height=self.config.height,
            image_width=self.config.width,
            R=R, T=t,
            FoVx=fovx, FoVy=fovy,
            uid=0
        )
        return cam

    @torch.no_grad()
    def _render_frame(self):
        """Render current frame"""
        self.config.T_limit = dpg.get_value("_TLimit")
        self.config.T_limit = -1 if self.config.T_limit == 0 else self.config.T_limit
        cam = self._build_camera()
        outputs = render(cam, self.gaussian_model, self.config, self.bg_color)
        
        rgb = outputs["render"].permute(1, 2, 0)  # (H, W, 3)
        rgb = torch.clamp(rgb, 0.0, 1.0)
        
        # Compose final image
        result = None
        num_layers = 0
        
        if self.render_rgb == "RGB":
            result = rgb.cpu().numpy()
            num_layers += 1
        elif self.render_rgb == "Surf-Depth":
            depth = outputs["surf_depth"].squeeze(0).unsqueeze(-1).expand_as(rgb)
            # notmalize depth to [0, 1] for visualization
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            result = depth.cpu().numpy()
            num_layers += 1
        elif self.render_rgb == "Dist":
            dist = outputs["rend_dist"].squeeze(0).unsqueeze(-1).expand_as(rgb)
            # normalize dist to [0, 1] for visualization
            dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-8)
            result = dist.cpu().numpy()
            num_layers += 1
        
        if result is None:
            result = rgb.cpu().numpy()
            num_layers = 1
        
        self.render_buffer = (result / num_layers).reshape(-1).astype(np.float32)
        dpg.set_value("_texture", self.render_buffer)

    def run(self):
        """Main render loop"""
        while dpg.is_dearpygui_running():
            self._render_frame()
            
            # Update FPS
            fps = self.fps_tracker.update()
            dpg.set_value("_fps", f"FPS: {fps:.1f}")
            
            dpg.render_dearpygui_frame()

    def __del__(self):
        dpg.destroy_context()


class FPSTracker:
    """Simple FPS tracker with smoothing"""
    
    def __init__(self, smoothing: float = 0.9):
        self.smoothing = smoothing
        self.last_time = time.time()
        self.fps = 30.0
    
    def update(self) -> float:
        current = time.time()
        dt = current - self.last_time
        self.last_time = current
        
        if dt > 0:
            instant_fps = 1.0 / dt
            self.fps = self.smoothing * self.fps + (1 - self.smoothing) * instant_fps
        
        return self.fps


def main():
    parser = ArgumentParser(description="3DGS Render Viewer")
    model_params = ModelParams(parser, sentinel=True)
    parser.add_argument('-l', '--load_iteration', type=int, default=30000, help="Iteration to load")
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    
    args = get_combined_args(parser)
    args.eval = False
    
    # Config
    config = RenderConfig(
        width=args.width,
        height=args.height,
        window_width=args.width,
        window_height=args.height,
        white_background=args.white_background,
        model_path=args.model_path,
        load_iteration=args.load_iteration,
        compute_cov3D_python=args.compute_cov3D_python,
        depth_ratio=args.depth_ratio
    )
    
    # Initialize model
    gs_model = GaussianModel(config.sh_degree)
    
    # Load scene to get cameras and determine iteration
    
    dataset = model_params.extract(args)
    scene = Scene(dataset, gs_model, load_iteration=config.load_iteration, shuffle=False)
    
    # Update config with actual loaded iteration
    config.load_iteration = scene.loaded_iter
    config.scene_pcd_path = f"{config.model_path}/point_cloud/iteration_{config.load_iteration}/point_cloud.ply"
    
    # Get training cameras
    data_cameras = scene.getTrainCameras()
    
    # Run GUI
    gui = GaussianRenderGUI(config, gs_model, data_cameras)
    gui.run()


if __name__ == "__main__":
    main()