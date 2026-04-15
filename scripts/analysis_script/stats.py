import os
from pathlib import Path
from adjustText import adjust_text
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from sklearn.metrics import silhouette_score, davies_bouldin_score
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

from matplotlib.colors import LogNorm
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from scene import Scene, GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from scipy import interpolate

class Interpolator:
    method_mapping = {
        "2DGS_dev_x0y0_pchip_33x2_fixed_cutoff_noedge_3sigma": "pchip.h",
        "2DGS_dev_pchip_x2_3sigma": "pchip.h",
        "2DGS_dev_x0y0_pchip3_33x2_fixed_cutoff_noedge_3sigma": "pchip3.h",
        "2DGS_dev_pchip3_x2_3sigma": "pchip3.h",
        "2DGS_dev_pchip3_x2_3sigma_rescale_fixed": "pchip3.h",
        "2DGS_dev_pchip3_x2_3sigma_rescale_grad_fixed": "pchip3_fixed.h",
        "2DGS_dev_pchip3_x2_3sigma_grad_fixed_inter_lr_0.0025": "pchip3_fixed.h",
        "2DGS_dev_x0y0_pchip_symmetric_33x2_fixed_cutoff_noedge_3sigma": "pchip_symmetric.h",
        "2DGS_dev_x0y0_pchip_symmetric_33x2_fixed_cutoff_noedge_3sigma_fixed": "pchip_symmetric.h",
        "2DGS_dev_pchip_symmetric_33x2_3sigma_so": "pchip_symmetric.h",
        "2DGS_dev_pchip_symmetric_33x2_3sigma_so_rhoo": "pchip_symmetric.h",
        "2DGS_dev_pchip_symmetric_33x2_3sigma_so_rhoo_fixed": "pchip_symmetric.h",
        "2DGS_dev_pchip5_x2_symmetric": "pchip5_symmetric.h",
        "2DGS_dev_pwl3_x2_3sigma": "pwl3.h",
        "2DGS_dev_pwl3_x2_inter_lr_0.001": "pwl3.h",
        "2DGS_dev_pwl3_x2_3sigma_grad_fixed": "pwl3.h",
        "2DGS_dev_pwl3_x2_3sigma_grad_fixed_inter_lr_0.0025": "pwl3.h",
        "2DGS_dev_pwl3_x2_3sigma_2grad_fixed_again_inter_lr_0.0025": "pwl3.h",
        "2DGS_dev_pwl3_y0_x2_3sigma": "pwl3_y0.h",
        "2DGS_dev_pwl_abs_x2_3sigma_inter_lr_0.0025": "pwl_abs.h",
        "2DGS_dev_pwl4_x2_3sigma": "pwl4.h",
        "2DGS_dev_pwl4_x2_3sigma_inter_lr_0.001": "pwl4.h",
        "2DGS_dev_pwl4_x2_3sigma_inter_lr_0.0025": "pwl4.h",
        "2DGS_dev_pwl5_symmetric_x2_3sigma": "pwl5_symmetric.h",
        "2DGS_dev_pwl5_asymmetric_x2_3sigma_inter_lr_0.0025": "pwl5_asymmetric.h",
        "2DGS_dev_pwl5_asymmetric_y0_x2_3sigma_inter_lr_0.0025": "pwl5_asymmetric_y0.h",
        "2DGS_dev_pwl4_to_pwl3_x2_3sigma": "pwl4.h",
        "2DGS_dev_pwl4_segment_x2_3sigma_bug_fixed": "pwl4_segment.h",
        "2DGS_dev_pwl4_segment_x2_3sigma_inter_lr_0.001": "pwl4_segment.h",
    }
    
    function_mapping = {
        "pwl3.h": None,
        "pwl4.h": None,
        "pwl4_accumulate.h": None,
        "pwl4_segment.h": None,
        "pwl5_symmetric.h": None,
        "pwl5_asymmetric.h": None,
        "pwl3_y0.h": None,
        "pwl5_asymmetric_y0.h": None,
        "pchip.h": None,
        "pchip3_fixed.h": None,
        "pwl_abs.h": None,
    }
    
    def __init__(self, dof, experiment, rescale=3):
        self.dof = dof
        self.method = self.method_mapping.get(experiment, None)
        if self.method is None:
            raise NotImplementedError(f"Experiment '{experiment}' is not recognized.")
        self.rescale = rescale
        
        self.x_ctrl = None
        self.y_ctrl = None
        self.ctrl_points = np.empty((0, 0, 0, 2))  # NxCxGx2, N: number, C: axis, G: control points, 2: (x,y)
    
    def pre_compute_v2(self, experiment, points):
        DIM = len(points.shape)
        if DIM == 1:  # just one group of point
            points = points[None, None, :]  # (1, 1, DIM)
        assert len(points.shape) == 3, "Expected points to be of shape (N, C, D)"
        N, C, D = points.shape
        def pwl3():
            """[x0, y0]->[[x0, y0]]"""
            assert D == 2, "Expected 2 parameters for pwl3 method"
            ctrl_points = points.reshape(N, C, 1, 2) * 2 - 1
            return ctrl_points
        def pwl4():
            """[x0, x1, y0, y1]->[[x0*x1, y0*y1], [x1, y1]]"""
            assert D == 4, "Expected 4 parameters for pwl4 method"
            p = points.reshape(N, C, 2, 2)  # [x0, x1, y0, y1] -> [[x0, x1], [y0, y1]]
            p = p.swapaxes(-1, -2)  # -> [[x0, y0], [x1, y1]]
            p[..., 0, 0] = p[..., 0, 0] * p[..., 1, 0]  # x0*x1, x1
            p[..., 0, 1] = p[..., 0, 1] * p[..., 1, 1]  # y0*y1, y1
            
            ctrl_points = p * 2 - 1  # rescale to [-1, 1]
            return ctrl_points
        def pwl4_accumulate():
            """[x0, x1, x2, y0, y1, y2]->[[x0, y0]/s, [x0+x1, y0+y1]/s]"""
            assert D == 6, "Expected 6 parameters for pwl4_accumulate method"
            p = points.reshape(N, C, 2, 3)  # [x0, x1, x2, y0, y1, y2] -> [[x0, x1, x2], [y0, y1, y2]]
            cumsum = np.cumsum(p, axis=-1)           # -> [[x0, y0], [x0+x1, y0+y1], [x0+x1+x2, y0+y1+y2]]
            total = cumsum[..., -1:]                  # -> [[x0+x1+x2, y0+y1+y2]]
            ctrl_points = (cumsum[..., :-1] / total * 2 - 1).swapaxes(-1, -2)  # -> [[x0/(x0+x1+x2), y0/(y0+y1+y2)], [(x0+x1)/(x0+x1+x2), (y0+y1)/(y0+y1+y2)]] -> rescale to [-1, 1]
            return ctrl_points
        def pwl4_segment():
            """[x0, x1, y0, y1]->[[-x0, y0*y1], [x1, y1]]"""
            assert D == 4, "Expected 4 parameters for pwl4_segment method"
            p = points.reshape(N, C, 2, 2)  # [x0, x1, y0, y1] -> [[x0, x1], [y0, y1]]
            p = p.swapaxes(-1, -2)  # -> [[x0, y0], [x1, y1]]
            x_ctrl = np.stack([-p[..., 0, 0], p[..., 0, 1]], axis=-1)
            y_ctrl = np.stack([p[..., 1, 0] * p[..., 1, 1], p[..., 1, 1]], axis=-1) * 2 - 1
            ctrl_points = np.stack([x_ctrl, y_ctrl], axis=-1)
            return ctrl_points
        def pwl5_symmetric():
            """[x0, y0]->[[-x0, -y0], [0, 0], [x0, y0]]"""
            assert D == 2, "Expected 2 parameters for pwl5_symmetric method"
            p = points.reshape(N, C, 1, 2)
            ctrl_points = np.concatenate([-p, np.zeros_like(p), p], axis=-2)  # (N, C, G, 2)
            return ctrl_points
        def pwl5_asymmetric():
            """[x0, y0, x1, y1]->[[-x0, -y0], [0, 0], [x1, y1]]"""
            assert D == 4, "Expected 4 parameters for pwl5_asymmetric method"
            p = points.reshape(N, C, 2, 2)  # [x0, y0, x1, y1] -> [[x0, y0], [x1, y1]]
            ctrl_points = np.concatenate([-p[..., :1, :], np.full_like(p[..., :1, :], 0), p[..., 1:, :]], axis=-2)  # (N, C, G, 2)
            return ctrl_points
        def pwl3_y0():
            """[x0]->[[x0, 0]]"""
            assert D == 1, "Expected 1 parameter for pwl3_y0 method"
            p = points.reshape(N, C, 1, 1)
            ctrl_points = np.concatenate([p * 2 - 1, np.zeros_like(p)], axis=-1) # (N, C, G, 2)
            return ctrl_points
        def pwl5_asymmetric_y0():
            """[x0, x1]->[[-x0, 0], [x1, 0]]"""
            assert D == 2, "Expected 2 parameters for pwl5_asymmetric_y0 method"
            p = points.reshape(N, C, 2, 1)
            p = np.stack([-p[..., 0, :], p[..., 1, :]], axis=-2)  # (N, C, G, 1)
            ctrl_points = np.concatenate([p, np.zeros_like(p)], axis=-1)  # (N, C, G, 2)
            return ctrl_points
        def pchip():
            """[x0, y0]->[[x0, y0]]
            0 < x0, y0 < 1
            """
            assert D == 2, "Expected 2 parameters for pchip method"
            ctrl_points = points.reshape(N, C, 1, 2)
            return ctrl_points
        def pwl_abs():
            assert D == 1, "Expected 2 parameters for pwl_abs method"
            p = points.reshape(N, C, 1, 1)
            a = p * 0.5
            xs = np.concatenate([a - 1, -a, np.zeros_like(a), a, 1 - a], axis=-2)  # [N, C, 5, 1]
            ys = np.array([-1., 0, 0, 0, 1.], dtype=np.float32)
            ys = ys.reshape(1, 1, 5, 1).repeat(N, axis=0).repeat(C, axis=1)
            ctrl_points = np.concatenate([xs, ys], axis=-1)  # [N, C, 5, 2]
            return ctrl_points
                        
        function_mapping = {
            "pwl3.h": pwl3,
            "pwl4.h": pwl4,
            "pwl4_accumulate.h": pwl4_accumulate,
            "pwl4_segment.h": pwl4_segment,
            "pwl5_symmetric.h": pwl5_symmetric,
            "pwl5_asymmetric.h": pwl5_asymmetric,
            "pwl3_y0.h": pwl3_y0,
            "pwl5_asymmetric_y0.h": pwl5_asymmetric_y0,
            "pchip.h": pchip,
            "pchip3_fixed.h": pwl3,
            "pwl_abs.h": pwl_abs,
        }
        method = self.method_mapping.get(experiment, None)
        fun = function_mapping.get(method, None)
        if fun is None:
            raise NotImplementedError(f"Experiment '{experiment}' is not recognized for pre-computation.")
        ctrl_points = fun()
        if DIM == 1:
            # append -1 and 1 for x and y
            ctrl_points = np.concatenate([np.full_like(ctrl_points[..., :1, :], -1), ctrl_points, np.full_like(ctrl_points[..., :1, :], 1)], axis=-2)
            ctrl_points = ctrl_points[0, 0]  # return to original shape if input was 1D
            self.x_ctrl = ctrl_points[..., 0]
            self.y_ctrl = ctrl_points[..., 1]
        else:
            self.ctrl_points = ctrl_points
        return self.ctrl_points
        
        
    def get_control_points(self, points: np.array):
        # points shape: (N, C, D), C: axis, D: dof
        N, C, D = points.shape
        if self.method in ["pwl3.h", "pchip3_fixed.h"]:
            assert D == 2, "Expected 2 parameters for pwl3 method"
            self.ctrl_points = points.reshape(N, C, 1, 2)  # (N, C, 1, 2)
            self.ctrl_points = self.ctrl_points * 2 - 1  # Rescale from [0, 1] to [-1, 1]
        elif self.method == "pwl4.h":
            assert D == 4, "Expected 4 parameters for pwl4 method"
            p = points.reshape(N, C, 2, 2)
            self.ctrl_points = np.stack([p[..., 0] * p[..., 1], p[..., 1]], axis=-1) * 2 - 1  # (N, C, G, 2)
        elif self.method == "pwl4_accumulate.h":
            assert D == 6, "Expected 6 parameters for pwl4_accumulate method"
            p = points.reshape(N, C, 2, 3)  # (N, C, G, 3)
            cumsum = np.cumsum(p, axis=-1)           # (N, C, 2, 3)
            total = cumsum[..., -1:]                  # (N, C, 2, 1)
            self.ctrl_points = (cumsum[..., :-1] / total * 2 - 1).swapaxes(-1, -2)  # (N, C, 2, G=2)
        elif self.method == "pwl4_segment.h":
            assert D == 4, "Expected 4 parameters for pwl4_segment method"
            p = points.reshape(N, C, 2, 2)
            x_ctrl = np.stack([-p[..., 0, 0], p[..., 0, 1]], axis=-1)
            y_ctrl = np.stack([p[..., 1, 0] * p[..., 1, 1], p[..., 1, 1]], axis=-1) * 2 - 1
            self.ctrl_points = np.stack([x_ctrl, y_ctrl], axis=-1)
        elif self.method == "pwl5_symmetric.h":
            assert D == 2, "Expected 2 parameters for pwl5_symmetric method"
            p = points.reshape(N, C, 1, 2)
            self.ctrl_points = np.concatenate([-p, p], axis=-2)  # (N, C, G, 2)
        elif self.method == "pwl5_asymmetric.h":
            assert D == 4, "Expected 4 parameters for pwl5_asymmetric method"
            p = points.reshape(N, C, 2, 2)
            self.ctrl_points = np.concatenate([-p[..., :1, :], p[..., 1:, :]], axis=-2)
        elif self.method == "pwl3_y0.h":
            assert D == 1, "Expected 1 parameter for pwl3_y0 method"
            p = points.reshape(N, C, 1, 1)
            self.ctrl_points = np.concatenate([p * 2 - 1, np.zeros_like(p)], axis=-1) # (N, C, G, 2)
        elif self.method == "pwl5_asymmetric_y0.h":
            assert D == 2, "Expected 2 parameters for pwl5_asymmetric_y0 method"
            p = points.reshape(N, C, 2, 1)
            p = np.stack([-p[..., 0, :], p[..., 1, :]], axis=-2)  # (N, C, G, 1)
            self.ctrl_points = np.concatenate([p, np.zeros_like(p)], axis=-1)  # (N, C, G, 2)
        elif self.method == "pchip.h":
            assert D == 2, "Expected 2 parameters for pchip method"
            p = points.reshape(N, C, 1, 2)
            self.ctrl_points = p
        else:
            raise NotImplementedError(f"Method '{self.method}' is not recognized.")
        return self.ctrl_points
    
    def pre_compute(self, points):
        if self.method == "pwl3.h":
            x0, y0 = points
            self.x_ctrl = np.array([0, x0, 1]) * 2 - 1
            self.y_ctrl = np.array([0, y0, 1]) * 2 - 1
        elif self.method == "pwl4.h":
            x0, x1, y0, y1 = points
            self.x_ctrl = np.array([0, x0 * x1, x1, 1]) * 2 - 1
            self.y_ctrl = np.array([0, y0 * y1, y1, 1]) * 2 - 1
        elif self.method == "pwl4_accumulate.h":
            x0, x1, x2, y0, y1, y2 = points
            sx = x0 + x1 + x2
            sy = y0 + y1 + y2
            self.x_ctrl = np.array([0, x0 / sx, (x0 + x1) / sx, 1]) * 2 - 1
            self.y_ctrl = np.array([0, y0 / sy, (y0 + y1) / sy, 1]) * 2 - 1
        elif self.method == "pwl4_segment.h":
            x0, x1, y0, y1 = points
            self.x_ctrl = np.array([-1, -x0, x1, 1])
            self.y_ctrl = np.array([0, y0 * y1, y1, 1]) * 2 - 1
        elif self.method == "pwl5_asymmetric.h":
            x0, y0, x1, y1 = points
            self.x_ctrl = np.array([-1, -x0, 0, x1, 1])
            self.y_ctrl = np.array([-1, -y0, 0, y1, 1])
        elif self.method == "pwl5_symmetric.h":
            x0, y0 = points
            self.x_ctrl = np.array([-1, -x0, 0, x0, 1])
            self.y_ctrl = np.array([-1, -y0, 0, y0, 1])
        elif self.method == "pchip3_fixed.h":
            x0, y0 = points
            self.x_ctrl = np.array([0, x0, 1]) * 2 - 1
            self.y_ctrl = np.array([0, y0, 1]) * 2 - 1
        elif self.method == "pwl3_y0.h":
            x0,  = points
            self.x_ctrl = np.array([0, x0, 1]) * 2 - 1
            self.y_ctrl = np.array([-1, 0, 1])
        elif self.method == "pwl5_asymmetric_y0.h":
            x0, x1 = points
            self.x_ctrl = np.array([-1, -x0, 0, x1, 1])
            self.y_ctrl = np.array([-1, 0, 0, 0, 1])
        elif self.method == "pchip.h":
            x0, y0 = points
            self.x_ctrl = np.array([-1, 0, x0, 1])
            self.y_ctrl = np.array([-1, 0, y0, 1])
        else:
            raise NotImplementedError(f"Method '{self.method}' is not recognized.")
        
    def eval(self, t):
        if "pwl" in self.method:
            t_deformed = np.interp(t / self.rescale, self.x_ctrl, self.y_ctrl) * self.rescale
        elif "pchip" in self.method:
            interpolator = interpolate.PchipInterpolator(self.x_ctrl, self.y_ctrl)
            t_deformed = interpolator(t / self.rescale) * self.rescale
        t_deformed[(t < -self.rescale) | (t > self.rescale)] = t[(t < -self.rescale) | (t > self.rescale)]
        return t_deformed

def gaussian(x):
    return np.exp(-0.5 * x ** 2)

def gaussian_cluster(gs_model: GaussianModel):
    inter_points = gs_model.get_inter_points.detach().cpu().numpy()
    scaling = gs_model.get_scaling.detach().cpu().numpy()
    opacity = gs_model.get_opacity.detach().cpu().numpy()
    
    inter_points_flat = inter_points.reshape(inter_points.shape[0], -1)
    features = np.concatenate([inter_points_flat, scaling, opacity], axis=-1)
    
    sample_indices = np.random.choice(features.shape[0], size=min(100000, features.shape[0]), replace=False)
    inter_points = inter_points[sample_indices]
    scaling = scaling[sample_indices]
    opacity = opacity[sample_indices]
    features = features[sample_indices]
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    
    # clusterer = KMeans(n_clusters=16, random_state=42, n_init=10)
    # labels = clusterer.fit_predict(features_scaled)
    
    clusterer = DBSCAN(eps=0.5, min_samples=10)
    labels = clusterer.fit_predict(features_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_scaled)
    fig, axes = plt.subplots(2, 3, figsize=(16, 16), constrained_layout=True)
    
    axes = axes.flatten()
    
    # (a) PCA 2D 可视化
    scatter = axes[0].scatter(features_2d[:, 0], features_2d[:, 1], 
                                  c=labels, cmap='tab10', s=1, alpha=0.6)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    axes[0].set_title(f'Cluster Visualization (PCA)')
    plt.colorbar(scatter, ax=axes[0], label='Cluster')
    
    # (b) 聚类大小分布
    unique, counts = np.unique(labels, return_counts=True)
    
    x0, y0 = inter_points[:, 0, 0], inter_points[:, 0, 1]
    scatter_b = axes[1].scatter(x0, y0, c=labels, cmap='tab10', s=1, alpha=0.6)
    axes[1].set_xlabel('x0')
    axes[1].set_ylabel('y0')
    axes[1].set_title('Inter Points Distribution (Colored by Cluster)')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter_b, ax=axes[1], label='Cluster ID')
    
    # (c) 特征重要性（各特征的方差）
    feature_names = [f'inter_{i}' for i in range(inter_points_flat.shape[1])] + \
                    [f'scale_{i}' for i in range(scaling.shape[1])] + \
                    [f'opacity_{i}' for i in range(opacity.shape[1])]
    
    feature_std = features.std(axis=0)
    top_n = min(20, len(feature_std))
    top_indices = np.argsort(feature_std)[-top_n:]
    axes[2].barh(range(top_n), feature_std[top_indices])
    axes[2].set_yticks(range(top_n))
    axes[2].set_yticklabels([feature_names[i] for i in top_indices], fontsize=8)
    axes[2].set_xlabel('Standard Deviation')
    axes[2].set_title(f'Top {top_n} Most Varying Features')
    axes[2].grid(True, alpha=0.3, axis='x')
    
    
    # (d) 每个聚类的平均 opacity 和 scaling
    cluster_stats = []
    for cluster_id in unique:
        mask = labels == cluster_id
        avg_opacity = opacity[mask].mean()
        avg_scale = scaling[mask].mean()
        cluster_stats.append([cluster_id, avg_opacity, avg_scale, counts[cluster_id == unique][0]])
    
    cluster_stats = np.array(cluster_stats)
    axes[3].scatter(cluster_stats[:, 1], cluster_stats[:, 2], 
                   s=cluster_stats[:, 3]/10, c=cluster_stats[:, 0], 
                   cmap='tab10', alpha=0.6)
    axes[3].set_xlabel('Average Opacity')
    axes[3].set_ylabel('Average Scaling')
    axes[3].set_title('Cluster Characteristics (size = count/10)')
    axes[3].grid(True, alpha=0.3)
    
    for i, cluster_id in enumerate(cluster_stats[:, 0]):
        axes[3].annotate(f'{int(cluster_id)}', 
                           (cluster_stats[i, 1], cluster_stats[i, 2]),
                           fontsize=8)
        
    axes[4].hist(opacity.flatten(), bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    axes[4].set_xlabel('Opacity Value')
    axes[4].set_ylabel('Frequency')
    axes[4].set_title('Opacity Distribution')
    axes[4].grid(True, alpha=0.3)
    
    plt.show()

def compute_deformed(i, coords_hd_sorted, TX_flat, TY_flat, tx, ty, 
                     features_scaling_opacity, with_scaling_and_opacity,
                     control_points, dof, experiment, rescaler):
    s = [TX_flat.copy(), TY_flat.copy()]
    t = [tx.copy(), ty.copy()]
    scaling = np.array([1,1]) if not with_scaling_and_opacity else features_scaling_opacity[i, :2]
    scaling = scaling / scaling.max()
    opacity = 1 if not with_scaling_and_opacity else features_scaling_opacity[i, 2]

    for ct in range(control_points):
        interpolator = Interpolator(dof=dof, experiment=experiment, rescale=rescaler)
        interpolator.pre_compute_v2(experiment, coords_hd_sorted[i, ct, :])
        # interpolator.pre_compute(coords_hd_sorted[i, ct, :])
        s[ct] = interpolator.eval(s[ct])
        t[ct] = interpolator.eval(t[ct])

    return s, t, scaling, opacity

def hist_2d(gs_model: GaussianModel, experiment, scene_name, file_paths):
    hist_path = file_paths["hist_path"]
    
    inter_points = gs_model.get_inter_points.detach().cpu().numpy()  # (N, C, D)
    
    D = gs_model.dof
    C = gs_model.control_points
    interpolator = Interpolator(dof=D, experiment=experiment, rescale=3)
    # ctrl_points = interpolator.get_control_points(inter_points)  # (N, C, G, 2)
    ctrl_points = interpolator.pre_compute_v2(experiment, inter_points)  # (N, C, G, 2)
    G = ctrl_points.shape[2]
    fig, axes = plt.subplots(1, 2, figsize=(20, 6), constrained_layout=True)
    title = f"Experiment: {experiment}\nScene: {scene_name}\n (N={inter_points.shape[0]}, C={C}, G={G})"
    fig.suptitle(title, fontsize=16, fontweight='bold')
    # use different color maps for different control points to distinguish them visually
    cmap = ["Reds", "Blues", "Greens", "Purples", "Oranges", "Greys"]
    for c in range(C):
        for g in range(G):
            x0, y0 = ctrl_points[:, c, g, 0], ctrl_points[:, c, g, 1]
            h = axes[c].hist2d(x=x0, y=y0, bins=[200, 200], cmap=cmap[(c*C+g) % len(cmap)], 
                               range=[[-1.1, 1.1], [-1.1, 1.1]],
                               norm=LogNorm(), cmin=1)
            h[3].set_alpha(0.5)
        axes[c].set_xlabel(f'Control Point {g} x (Axis {c})')
        # set axis ratio to 1 to make the plot square
        axes[c].set_aspect('equal', adjustable='box')
        axes[c].set_title(f'2D Histogram of Control Points (Axis {c})')
    plt.savefig(hist_path, dpi=450)
    plt.close(fig)
    # features = inter_points.reshape(-1, dof * control_points)  # (N, dof * control_points)
    
def kernel_cluster(gs_model: GaussianModel, experiment, scene_name, file_paths):
    deformation_path = file_paths["deformation_path"]
    
    inter_points = gs_model.get_inter_points.detach().cpu().numpy()  # (N, C, D)
    scalings = gs_model.get_scaling.detach().cpu().numpy()
    opacities = gs_model.get_opacity.detach().cpu().numpy()
    D = gs_model.dof
    C = gs_model.control_points
    S = 1000 # sample size
    interpolator = Interpolator(dof=D, experiment=experiment, rescale=3)
    # Sample a subset of points for cluster analysis to reduce computational load
    sample_indices = np.random.choice(inter_points.shape[0], size=min(S, inter_points.shape[0]), replace=False)
    inter_points_sampled = inter_points[sample_indices]
    scaling_sampled = scalings[sample_indices]
    opacity_sampled = opacities[sample_indices]
    
    ctrl_points_sampled = interpolator.pre_compute_v2(experiment, inter_points_sampled)  # (S, C, G, 2)
    
    # [N, C, G, 2] -> [N, C, G*2] -> [N, C*G*2]
    ctrl_points = np.concatenate([np.full_like(ctrl_points_sampled[..., :1, :], -1), ctrl_points_sampled, np.full_like(ctrl_points_sampled[..., :1, :], 1)], axis=-2)
    
    
    R = 20
    LIMIT = 3.6
    tx = np.linspace(-LIMIT, LIMIT, R)
    # ty = np.linspace(-LIMIT, LIMIT, R)
    
    # TX, TY = np.meshgrid(tx, ty)  # (R, R)
    # TX_flat = TX.flatten()  # (R*R,)
    # TY_flat = TY.flatten()  # (R*R,)
    C = ctrl_points_sampled.shape[1]
    # t = np.stack([tx, ty], axis=0)  # (2, R)
    
    def cluster(curves, n_clusters=16):
        surfaces_flat = np.array(curves)  # (S, R*R)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(surfaces_flat)
        return labels
    
    surfaces_with_so = []
    surfaces_without_so = []
    curves = []
    for s in range(S):
        curve = []
        for c in range(C):
            interpolator.x_ctrl = ctrl_points[s, c, :, 0]  # [-1, ..., 1]
            interpolator.y_ctrl = ctrl_points[s, c, :, 1]  # [-1, ..., 1]
            curve.append(interpolator.eval(tx))
        curve = np.array(curve).reshape(-1)  # (C*R)  [tx1, tx2, ..., ty1, ty2, ] for all control points
        curves.append(curve)
    curves = np.array(curves)  # (S, C*R)
    n_clusters = 16
    labels = cluster(curves, n_clusters=n_clusters)

    # palettes = ['#4878d0', '#ee854a', '#6acc64', '#d65f5f', '#956cb4',  
    #             '#8c613c', '#dc7ec0', '#797979', '#d5bb67', '#82c6e2',  
    #             '#2e5aa8', '#d66b28', '#4ab345', '#c03a3a', '#7a4e9d',
    #             '#6f4623', '#ca5aa9', '#5a5a5a', '#c4a73f', '#5bb0d5',
    #             ]
    PLOT_SAMPLE_SIZE = 300
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    title = f"Clustered Deformation Curves\nExperiment: {experiment}, Scene: {scene_name}"
    fig.suptitle(title, fontsize=16, fontweight='bold')
    # idx = np.random.choice(S, min(PLOT_SAMPLE_SIZE, S), replace=False)
    # for i in idx:
    #     c = palettes[labels[i] % len(palettes)]
    #     axes[0].plot(tx, curves[i][:R], color=c, alpha=0.08, linewidth=0.6, zorder=2)
    #     axes[1].plot(tx, curves[i][R:], color=c, alpha=0.08, linewidth=0.6, zorder=2)

    cmap = plt.get_cmap('rainbow')
    total = len(labels)
    order = sorted(range(n_clusters), key=lambda k: (labels == k).sum(), reverse=True)
    color_map = {k: cmap(rank / n_clusters) for rank, k in enumerate(order)}
    idx = np.random.choice(S, min(PLOT_SAMPLE_SIZE, S), replace=False)
    for i in idx:
        c = color_map[labels[i]]
        axes[0].plot(tx, curves[i][:R], color=c, alpha=0.08, linewidth=0.6, zorder=2)
        axes[1].plot(tx, curves[i][R:], color=c, alpha=0.08, linewidth=0.6, zorder=2)
        
    # 每簇的均值曲线
    for k in order:
        mask = labels == k
        if mask.sum() == 0:
            continue
        mean_y = curves[mask].mean(axis=0)
        c = color_map[k]  
        pct = mask.sum() / total * 100
        axes[0].plot(tx, mean_y[:R], color=c, linewidth=1, zorder=5,
                label=f'({pct:.1f}%)')
        axes[1].plot(tx, mean_y[R:], color=c, linewidth=1, zorder=5,
                label=f'({pct:.1f}%)')
    
    axes[0].set_aspect('equal', adjustable='box')
    axes[1].set_aspect('equal', adjustable='box')
    axes[0].set_xlim(-LIMIT, LIMIT)
    axes[1].set_xlim(-LIMIT, LIMIT)

    # Add horizontal legend at the top of the figure
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc='center right',
               ncol=1, frameon=False,
               bbox_to_anchor=(1.01, 0.5), fontsize=9)
    plt.savefig(deformation_path, dpi=450)
    plt.close(fig)
        # ## with scaling and opacity
        # scaling = scaling_sampled[s]
        # scaling = scaling / scaling.max()
        # opacity = opacity_sampled[s]
        # TX, TY = np.meshgrid(tx / scaling[0], ty / scaling[1])  # (R, R)
        # TX_flat = TX.flatten()  # (R*R,)
        # TY_flat = TY.flatten()  # (R*R,)

        # E = np.linalg.norm([TX_flat, TY_flat], axis=0).reshape(R, R)  # (R, R)
        # G = gaussian(E) * opacity
        # surfaces_with_so.append(G)
        
        # TX, TY = np.meshgrid(tx, ty)  # (R, R)
        # TX_flat = TX.flatten()  # (R*R,)
        # TY_flat = TY.flatten()  # (R*R,)

        # E = np.linalg.norm([TX_flat, TY_flat], axis=0).reshape(R, R)  # (R, R)
        # G = gaussian(E)
        # surfaces_without_so.append(G)
        

    
    # surfaces_with_so = np.array(surfaces_with_so)  # (S, R, R)
    # surfaces_without_so = np.array(surfaces_without_so)  # (S, R, R)
    
    # labels_with_so = cluster(surfaces_with_so)
    # labels_without_so = cluster(surfaces_without_so)
    
    

def find_peak(gs_model: GaussianModel, experiment, file_paths, num_peaks=16, with_scaling_and_opacity=False):    
    pca_peak_path = file_paths.get("pca_peak_path", None)
    contourf_path = file_paths.get("contourf_path", None)
    deformed_path = file_paths.get("deformed_path", None)
    scaled_path = file_paths.get("scaled_path", None)
    
    inter_points = gs_model.get_inter_points.detach().cpu().numpy()
    scaling = gs_model.get_scaling.detach().cpu().numpy()
    opacity = gs_model.get_opacity.detach().cpu().numpy()
    
    dof = gs_model.dof
    control_points = gs_model.control_points
    features = inter_points.reshape(-1, dof * control_points)  # (N, dof * control_points)
    if with_scaling_and_opacity:
        features = np.column_stack([features, scaling, opacity])
    
    # PCA降维
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(features)

    # 2D直方图和峰值检测
    counts_2d, xedges, yedges = np.histogram2d(data_2d[:, 0], data_2d[:, 1], bins=200)
    counts_smooth = gaussian_filter(counts_2d, sigma=2)
    peaks_2d = peak_local_max(counts_smooth, min_distance=1, num_peaks=num_peaks, exclude_border=False)

    # 提取峰值高度并排序
    peak_heights = counts_smooth[peaks_2d[:, 0], peaks_2d[:, 1]]
    sorted_indices = np.argsort(peak_heights)[::-1]
    peaks_2d_sorted = peaks_2d[sorted_indices]
    peak_heights_sorted = peak_heights[sorted_indices]

    # 提取排序后的PC坐标
    peak_pcs_sorted = np.column_stack([
        xedges[peaks_2d_sorted[:, 0]],
        yedges[peaks_2d_sorted[:, 1]]
    ])

    # 逆变换到高维空间
    coords_hd_sorted = pca.inverse_transform(peak_pcs_sorted)

    # 打印峰值信息
    print(f"找到 {len(coords_hd_sorted)} 个峰值（按高度排序）:")

    if pca_peak_path is not None:
        # ========== 可视化 PCA空间 ==========
        fig = plt.figure(figsize=(10, 8), constrained_layout=True)
        ax = plt.gca()
        
        # 背景等高线
        plt.contourf(xedges[:-1], yedges[:-1], counts_smooth.T, 
                    levels=20, cmap='viridis', alpha=0.8)

        # 峰值标记点
        plt.scatter(peak_pcs_sorted[:, 0], peak_pcs_sorted[:, 1], 
                c='yellow', s=400, marker='*', edgecolors='black', 
                linewidths=2.5, zorder=10)

        # 🔥 文本 + 手动箭头
        x_range = xedges.max() - xedges.min()
        y_range = yedges.max() - yedges.min()
        offset_scale = 0.15

        for i in range(len(peak_pcs_sorted)):
            px, py = peak_pcs_sorted[i, 0], peak_pcs_sorted[i, 1]  # 峰值点
            
            angle = (i * 137.5) % 360
            dx = offset_scale * x_range * np.cos(np.radians(angle))
            dy = offset_scale * y_range * np.sin(np.radians(angle))
            tx, ty = px + dx, py + dy  # 文本位置
            
            # 画箭头：从文本指向峰值点
            arrow = FancyArrowPatch(
                (tx, ty), (px, py),
                arrowstyle='->',
                color='#404040',
                linewidth=1.5,
                alpha=0.8,
                shrinkA=15,          # 从文本边缘开始
                shrinkB=5,           # 到点附近结束
                mutation_scale=12,
                zorder=9
            )
            ax.add_patch(arrow)
            
            # 画文本
            ax.text(tx, ty, f'{i+1}',
                    fontsize=13, 
                    fontweight='bold',
                    color='black',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.4', 
                            facecolor='white', 
                            edgecolor='black', 
                            linewidth=1.5, 
                            alpha=0.95),
                    zorder=11)

        plt.xlabel('PC1', fontsize=12)
        plt.ylabel('PC2', fontsize=12)
        plt.title('Peaks in PCA Space (sorted by height)', fontsize=14)
        plt.colorbar(label='Density')
        plt.savefig(pca_peak_path, dpi=150)
        plt.close(fig)
    

    
    tx = np.linspace(-3.6, 3.6, 100)
    ty = np.linspace(-3.6, 3.6, 100)
    
    TX, TY = np.meshgrid(tx, ty)
    TX_flat = TX.flatten()
    TY_flat = TY.flatten()
    
    row = np.ceil(np.sqrt(num_peaks)).astype(int)
    col = np.ceil(num_peaks / row).astype(int)
    coords_hd_sorted = np.clip(coords_hd_sorted, 0.0001, 0.9999)
    features_inter_points = coords_hd_sorted[:, :dof*control_points]
    features_scaling_opacity = coords_hd_sorted[:, dof*control_points:]
    coords_hd_sorted = features_inter_points.reshape(-1, control_points, dof)
    rescaler = 3
    
    # plot the deformation contour for each peak
    if contourf_path is not None:
        fig1, axes1 = plt.subplots(row, col, figsize=(4*col, 4*row), constrained_layout=True)
        axes = axes1.flatten()
        for i in range(len(coords_hd_sorted)):
            ax = axes[i]
            s, t, scaling, opacity = compute_deformed(i, coords_hd_sorted, TX_flat, TY_flat, tx, ty,
                                                    features_scaling_opacity, with_scaling_and_opacity,
                                                    control_points, dof, experiment, rescaler)
            e = np.linalg.norm([s[0].reshape(TX.shape) / scaling[0], s[1].reshape(TY.shape) / scaling[1]], axis=0)
            G = gaussian(e) * opacity
            ax.contourf(TX, TY, G, levels=20, cmap='viridis')
            ax.set_title(f'Peak {i+1}, count: {peak_heights_sorted[i]:.2f}', fontsize=12)
            ax.set_xlim(-3.6, 3.6)
            ax.set_ylim(-3.6, 3.6)
        fig1.savefig(contourf_path, dpi=450)
        plt.close(fig1)
    if deformed_path is not None:
        fig2, axes2 = plt.subplots(row, col, figsize=(4*col, 4*row), constrained_layout=True)
        axes = axes2.flatten()
        for i in range(len(coords_hd_sorted)):
            ax = axes[i]
            s, t, scaling, opacity = compute_deformed(i, coords_hd_sorted, TX_flat, TY_flat, tx, ty,
                                                    features_scaling_opacity, with_scaling_and_opacity,
                                                    control_points, dof, experiment, rescaler)
            ax.plot(tx, t[0], alpha=0.7)
            ax.plot(ty, t[1], alpha=0.7)
            ax.set_title(f'Peak {i+1}, count: {peak_heights_sorted[i]:.2f}', fontsize=12)
            ax.set_xlim(-3.6, 3.6)
            ax.set_ylim(-3.6, 3.6)
        fig2.savefig(deformed_path, dpi=150)
        plt.close(fig2)
    if scaled_path is not None:
        fig3, axes3 = plt.subplots(row, col, figsize=(4*col, 4*row), constrained_layout=True)
        axes = axes3.flatten()
        for i in range(len(coords_hd_sorted)):
            ax = axes[i]
            s, t, scaling, opacity = compute_deformed(i, coords_hd_sorted, TX_flat, TY_flat, tx, ty,
                                                features_scaling_opacity, with_scaling_and_opacity,
                                                control_points, dof, experiment, rescaler)
            ax.plot(tx, gaussian(t[0] / scaling[0]) * opacity, alpha=0.7)
            ax.plot(ty, gaussian(t[1] / scaling[1]) * opacity, alpha=0.7)
            ax.set_title(f'Peak {i+1}, count: {peak_heights_sorted[i]:.2f}', fontsize=12)
            ax.set_xlim(-3.6, 3.6)
            ax.set_ylim(-3.6, 3.6)
        fig3.savefig(scaled_path, dpi=150)
        plt.close(fig3)

    
def gaussian_shape_analysis(gs_model: GaussianModel):
    sigma = 2
    inter_points = gs_model.get_inter_points.detach().cpu().numpy()
    
    x0, y0 = inter_points[:, 0, 0], inter_points[:, 0, 1]
    x1, y1 = inter_points[:, 1, 0], inter_points[:, 1, 1]
    t = np.linspace(-3.6, 3.6, 100)
    n_points = len(x0)
    
    counts_0, xedges_0, yedges_0 = np.histogram2d(x0, y0, bins=[200, 200])
    counts_1, xedges_1, yedges_1 = np.histogram2d(x1, y1, bins=[200, 200])
    counts_smooth_0 = gaussian_filter(counts_0, sigma=sigma)  # sigma调大 = 更平滑
    counts_smooth_1 = gaussian_filter(counts_1, sigma=sigma)  # sigma调大 = 更平滑
    
    
    
    
    peaks = peak_local_max(counts_smooth_0, 
                           min_distance=1,      
                           exclude_border=False,
                           num_peaks=9)
    
    peak_coords = []
    peak_heights = []
    for peak in peaks:
        i, j = peak
        t_val = xedges_0[i]
        x_val = yedges_0[j]
        height = counts_smooth_0[i, j]
        peak_coords.append((t_val, x_val))
        peak_heights.append(height)
        
    sorted_indices = np.argsort(peak_heights)[::-1]
    peak_coords = [peak_coords[i] for i in sorted_indices]
    peak_heights = [peak_heights[i] for i in sorted_indices]

    fig = plt.figure(figsize=(20, 6), constrained_layout=True)
    
    ax1 = fig.add_subplot(131)
    h = ax1.hist2d(x=x0, y=y0, bins=[200, 200], 
                cmap='Blues', norm=LogNorm(), cmin=1)
    # 标注峰值
    for i, (t, x) in enumerate(peak_coords):  # 只标注前5个
        ax1.plot(t, x, 'r*', markersize=20, markeredgecolor='white', markeredgewidth=2)
        ax1.text(t, x, f'{i+1}', color='red', fontsize=12, 
                ha='center', va='bottom', fontweight='bold')
    ax1.set_xlabel('t')
    ax1.set_ylabel('gaussian_out_x')
    ax1.set_title('2D Histogram with Peaks')
    plt.colorbar(h[3], ax=ax1)
    
    ax2 = fig.add_subplot(132, projection='3d')
    xpos, ypos = np.meshgrid(xedges_0[:-1], yedges_0[:-1], indexing="ij")
    surf = ax2.plot_surface(xpos, ypos, counts_smooth_0, cmap='Blues', 
                            alpha=0.7, edgecolor='none')
    # 标注峰值
    peak_t = [p[0] for p in peak_coords]
    peak_x = [p[1] for p in peak_coords]
    peak_h = peak_heights
    ax2.scatter(peak_t, peak_x, peak_h, c='red', s=200, marker='*', 
                edgecolors='white', linewidths=2, zorder=10)
    ax2.set_xlabel('t')
    ax2.set_ylabel('gaussian_out_x')
    ax2.set_zlabel('Count (smoothed)')
    ax2.set_title('Smoothed 3D Surface')
    # ax2.view_init(elev=30, azim=45)

    # 右：等高线图 + 峰值
    ax3 = fig.add_subplot(133)
    levels = np.logspace(np.log10(counts_smooth_0[counts_smooth_0>0].min()), 
                        np.log10(counts_smooth_0.max()), 20)
    cs = ax3.contourf(xedges_0[:-1], yedges_0[:-1], counts_smooth_0.T, 
                    levels=levels, cmap='Blues', norm=LogNorm())
    ax3.contour(xedges_0[:-1], yedges_0[:-1], counts_smooth_0.T, 
                levels=levels, colors='white', alpha=0.3, linewidths=0.5)
    # 标注峰值
    for i, (t, x) in enumerate(peak_coords):
        ax3.plot(t, x, 'r*', markersize=20, markeredgecolor='white', markeredgewidth=2)
        ax3.text(t, x, f'{i+1}', color='red', fontsize=12, 
                ha='right', va='top', fontweight='bold')
    ax3.set_xlabel('t')
    ax3.set_ylabel('gaussian_out_x')
    ax3.set_title('Contour with Peaks')
    plt.colorbar(cs, ax=ax3)
    
    plt.show()


def inter_points_stats(gs_model: GaussianModel):
    inter_points = gs_model.get_inter_points.detach().cpu().numpy()
    
    x0, y0 = inter_points[:, 0, 0], inter_points[:, 0, 1]
    x1, y1 = inter_points[:, 1, 0], inter_points[:, 1, 1]
    
    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    ax = fig.add_subplot(121)    

    
    h = ax.hist2d(x0, y0, bins=200, cmap='hot', norm=LogNorm())
    ax.set_xlabel('x0')
    ax.set_ylabel('y0')
    ax.set_title('Density Heatmap of Inter Points (x0, y0)')
    ax.grid(True, alpha=0.3)
    ax = fig.add_subplot(122)
        
    h = ax.hist2d(x1, y1, bins=200, cmap='hot', norm=LogNorm())
    ax.set_xlabel('x1')
    ax.set_ylabel('y1')
    ax.set_title('Density Heatmap of Inter Points (x1, y1)')
    ax.grid(True, alpha=0.3)
    plt.show()
    
    # x0_ctrl = np.concatenate([-np.ones_like(x0), np.zeros_like(x0), x0, np.ones_like(x0)])
    # y0_ctrl = np.concatenate([-np.ones_like(y0), np.zeros_like(y0), y0, np.ones_like(y0)])
    # x1_ctrl = np.concatenate([-np.ones_like(x1), np.zeros_like(x1), x1, np.ones_like(x1)])
    # y1_ctrl = np.concatenate([-np.ones_like(y1), np.zeros_like(y1), y1, np.ones_like(y1)])
    
    # tx = np.linspace(-3.6, 3.6, 100)
    # ty = np.linspace(-3.6, 3.6, 100)
    # TX, TY = np.meshgrid(tx, ty)
    # t = np.concatenate([TX.flatten(), TY.flatten()])
    t = np.linspace(-3.6, 3.6, 100)

    # 对每个点进行插值
    n_points = len(x0)
    out_x = np.zeros((n_points, len(t)))
    out_y = np.zeros((n_points, len(t)))
    gaussian_out_x = np.zeros((n_points, len(t)))
    gaussian_out_y = np.zeros((n_points, len(t)))

    for i in tqdm(range(0, n_points, 1000)):
        # 为每个点创建控制点
        x0_ctrl = np.array([-1, 0, x0[i], 1])
        y0_ctrl = np.array([-1, 0, y0[i], 1])
        x1_ctrl = np.array([-1, 0, x1[i], 1])
        y1_ctrl = np.array([-1, 0, y1[i], 1])
        
        # 插值
        out_x[i] = interpolate.pchip_interpolate(x0_ctrl, y0_ctrl, t / 3) * 3
        out_y[i] = interpolate.pchip_interpolate(x1_ctrl, y1_ctrl, t / 3) * 3
        
        # 修正2: 边界条件处理
        mask = (t < -3.0) | (t > 3.0)
        out_x[i, mask] = t[mask]
        out_y[i, mask] = t[mask]
        gaussian_out_x[i] = gaussian(out_x[i])
        gaussian_out_y[i] = gaussian(out_y[i])

    t_flat = np.tile(t, n_points)  # 重复 t 值
    out_x_flat = out_x.flatten()       # 展平所有输出值
    out_y_flat = out_y.flatten()       # 展平所有输出值
    gaussian_out_x_flat = gaussian_out_x.flatten()  # 展平高斯输出值
    gaussian_out_y_flat = gaussian_out_y.flatten()  # 展平高斯输出值
    fig, axs = plt.subplots(2, 2, figsize=(16, 6), constrained_layout=True)
    ax1, ax2 = axs[0, 0], axs[0, 1]
    # 绘制2D直方图密度
    
    ax1.hist2d(t_flat, out_x_flat, bins=[200, 200], cmap='hot',  
               norm=LogNorm(), cmin=1)  # 使用 LogNorm
    fig.colorbar(ax1.collections[0], ax=ax1, label='Curve Density (log scale)')
    ax1.set_xlabel('t value')
    ax1.set_ylabel('Output value')
    ax1.set_title(f'Curve Density Heatmap (Log Scale) ({n_points} curves)')
    ax1.grid(True, alpha=0.3)

    # 绘制高斯函数曲线
    ax2.hist2d(x=t_flat, y=gaussian_out_x_flat, bins=[200, 200], cmap='Blues',
               norm=LogNorm(), cmin=1)  # 使用 LogNorm
    fig.colorbar(ax2.collections[0], ax=ax2, label='Gaussian Value Density (log scale)')
    ax2.set_xlabel('t value')
    ax2.set_ylabel('Gaussian Value')
    ax2.set_title(f'Gaussian Function Curve ({n_points} curves)')
    ax2.grid(True, alpha=0.3)
    
    ax1, ax2 = axs[1, 0], axs[1, 1]
    
    ax1.hist2d(t_flat, out_y_flat, bins=[200, 200], cmap='hot',  
               norm=LogNorm(), cmin=1)  # 使用 LogNorm
    fig.colorbar(ax1.collections[0], ax=ax1, label='Curve Density (log scale)')
    ax1.set_xlabel('t value')
    ax1.set_ylabel('Output value')
    ax1.set_title(f'Curve Density Heatmap (Log Scale) ({n_points} curves)')
    ax1.grid(True, alpha=0.3)

    # 绘制高斯函数曲线
    ax2.hist2d(x=t_flat, y=gaussian_out_y_flat, bins=[200, 200], cmap='Blues',
               norm=LogNorm(), cmin=1)  # 使用 LogNorm
    fig.colorbar(ax2.collections[0], ax=ax2, label='Gaussian Value Density (log scale)')
    ax2.set_xlabel('t value')
    ax2.set_ylabel('Gaussian Value')
    ax2.set_title(f'Gaussian Function Curve ({n_points} curves)')
    ax2.grid(True, alpha=0.3)

    plt.show()


def file_paths(experiment, scene, with_scaling_and_opacity):
    output_dir = Path(f'scripts/analysis_script/output') / experiment / scene
    os.makedirs(output_dir, exist_ok=True)
    pca_peak_path = output_dir / f'scaling_and_opacity_{with_scaling_and_opacity}_pca_peaks.pdf'
    contourf_path = output_dir / f'scaling_and_opacity_{with_scaling_and_opacity}_contourf.pdf'
    deformed_path = output_dir / f'scaling_and_opacity_{with_scaling_and_opacity}_deformation_function.pdf'
    scaled_path = output_dir / f'scaling_and_opacity_{with_scaling_and_opacity}_gaussian_scaled.pdf'
    hist_path = output_dir / f'histogram_2d.pdf'
    deformation_path = output_dir / f'kernel_cluster.pdf'
    
    file_map = {"pca_peak_path": pca_peak_path,
            "contourf_path": contourf_path,
            "deformed_path": deformed_path,
            "scaled_path": scaled_path,
            "hist_path": hist_path,
            "deformation_path": deformation_path, }
    
    non_exists_files = dict([(k, v) for k, v in file_map.items() if not v.exists()])
    if len(non_exists_files) != 0:
        print(f"以下分析结果不存在，将进行分析:")
        for k, v in non_exists_files.items():
            print(f"  - {k}: {v}")

    return non_exists_files

def main():
    
    # Initialize model
    gs_path = Path("/mnt/cviss/Matt/GS-Output")
    model_paths = list(gs_path.glob("2DGS_dev*"))
    for model_path in model_paths:
        experiment = model_path.parts[-1]
        if experiment not in Interpolator.method_mapping:
            # print(f"实验 '{experiment}' 不在方法映射中，跳过分析。")
            continue
        method = Interpolator.method_mapping[experiment]
        if method not in Interpolator.function_mapping:
            print(f"实验 '{experiment}' 的方法 '{method}' 没有对应的函数实现，跳过分析。")
            continue
        if experiment != "2DGS_dev_pwl5_asymmetric_x2_3sigma_inter_lr_0.0025":
            continue
        print(f"分析模型: {experiment}")
        for scene in model_path.glob("Mip-NeRF360/360_v2/*/"):
            try:
                scene_name = "/".join(scene.parts[-3:])
                print(f"  场景: {scene_name}")
                file_paths_with_so = file_paths(experiment, scene_name, with_scaling_and_opacity=True)
                file_paths_without_so = file_paths(experiment, scene_name, with_scaling_and_opacity=False)
                if len(file_paths_with_so) == 0 or len(file_paths_without_so) == 0:
                    continue  # 如果所有文件都存在，跳过分析
                
                gs_model = GaussianModel(3)
                gs_model.load_ply(scene / "point_cloud" / f"iteration_30000" / "point_cloud.ply")
                # /mnt/cviss/Matt/GS-Output/2DGS_dev_pwl5_symmetric_3sigma/Mip-NeRF360/360_v2/bicycle_4
                # inter_points_stats(gs_model)
                # gaussian_shape_analysis(gs_model)
                # gaussian_cluster(gs_model)
                if len(file_paths_with_so) > 0:
                    find_peak(gs_model, experiment, file_paths_with_so, with_scaling_and_opacity=True)
                if len(file_paths_without_so) > 0:
                    find_peak(gs_model, experiment, file_paths_without_so, with_scaling_and_opacity=False)
                if "hist_path" in file_paths_with_so:
                    hist_2d(gs_model, experiment, scene_name, file_paths_with_so)
                if "deformation_path" in file_paths_with_so:
                    kernel_cluster(gs_model, experiment, scene_name, file_paths_with_so)
            except NotImplementedError as e:
                print(f"跳过实验 '{experiment}'，场景 '{scene_name}'，原因: {e}")
            except FileNotFoundError as e:
                print(f"文件未找到，跳过实验 '{experiment}'，场景 '{scene_name}'，原因: {e}")
            # except Exception as e:
            #     print(f"分析过程中发生错误，跳过实验 '{experiment}'，场景 '{scene_name}'，原因: {e}")
        #     break
        # break

if __name__ == "__main__":
    main()