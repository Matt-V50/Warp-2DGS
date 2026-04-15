import os
from adjustText import adjust_text
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from sklearn.metrics import silhouette_score, davies_bouldin_score
from matplotlib.patches import FancyArrowPatch


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

def find_peak(gs_model: GaussianModel, num_peaks=16):
    inter_points = gs_model.get_inter_points.detach().cpu().numpy()
    dof = gs_model.dof
    control_points = gs_model.control_points
    data_hd = inter_points.reshape(-1, dof * control_points)  # (N, dof * control_points)
    
    # PCA降维
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data_hd)

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
    plt.show()

    
    tx = np.linspace(-3.6, 3.6, 100)
    ty = np.linspace(-3.6, 3.6, 100)
    
    TX, TY = np.meshgrid(tx, ty)
    TX_flat = TX.flatten()
    TY_flat = TY.flatten()
    
    r = np.ceil(np.sqrt(num_peaks)).astype(int)
    c = np.ceil(num_peaks / r).astype(int)
    fig, axes = plt.subplots(r, c, figsize=(4*c, 4*r), constrained_layout=True)
    axes = axes.flatten()
    coords_hd_sorted = np.clip(coords_hd_sorted, 0.0001, 0.9999)
    coords_hd_sorted = coords_hd_sorted.reshape(-1, gs_model.control_points, gs_model.dof)
    rescaler = 3
    for i in range(len(coords_hd_sorted)):
        ax = axes[i]
        s = [TX_flat, TY_flat]
        for c in range(control_points):
            if dof == 6:
                p_x0, p_x1, p_x2, p_y0, p_y1, p_y2 = coords_hd_sorted[i, c, :]
                sum_x = p_x0 + p_x1 + p_x2
                sum_y = p_y0 + p_y1 + p_y2
                x_ctrl = np.array([0, p_x0 / sum_x, (p_x1 + p_x2) / sum_x, 1]) * 2 - 1
                y_ctrl = np.array([0, p_y0 / sum_y, (p_y1 + p_y2) / sum_y, 1]) * 2 - 1
            s_deformed = np.interp(s[c] / rescaler, x_ctrl, y_ctrl) * rescaler
            s_deformed[(s[c] < -3) | (s[c] > 3)] = s[c][(s[c] < -3) | (s[c] > 3)]
            s[c] = s_deformed

        e = np.linalg.norm([s[0].reshape(TX.shape), s[1].reshape(TY.shape)], axis=0)
        G = gaussian(e)
        
        # G = g.reshape(TX.shape)
        ax.contourf(TX, TY, G, levels=20, cmap='viridis')
        
        ax.set_title(f'Peak {i+1}, count: {peak_heights_sorted[i]:.2f}', fontsize=12)
        ax.set_xlim(-3.6, 3.6)
        ax.set_ylim(-3.6, 3.6)
        
    plt.show()
    
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


def main():
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    args = get_combined_args(parser)
    args.eval = False
    
    dataset, opt, pipe = lp.extract(args), op.extract(args), pp.extract(args)
    
    # Initialize model
    gs_model = GaussianModel(dataset.sh_degree)
    gs_model.load_ply(os.path.join(dataset.model_path, "point_cloud", "iteration_" + str(opt.iterations), "point_cloud.ply"))
    
    # inter_points_stats(gs_model)
    # gaussian_shape_analysis(gs_model)
    # gaussian_cluster(gs_model)
    find_peak(gs_model)

if __name__ == "__main__":
    main()