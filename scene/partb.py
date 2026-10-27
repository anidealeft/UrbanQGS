import os
import numpy as np
import collections
import struct
import argparse
from scipy.spatial import cKDTree
import open3d as o3d
from tqdm import tqdm
import shutil

import matplotlib.pyplot as plt
import matplotlib.patches as patches
# 引入你提供的 loader (假设文件名是 colmap_loader.py)
import colmap_loader as colmap_utils


def visualize_overlaps_only(points_xyz, active_block_data, output_path):
    """
    只画出重叠点（Overlap Points）。
    如果两个 Block 之间有渗透，重叠区域会非常明显。
    """
    print("Generating OVERLAP ONLY visualization...")

    # 1. 统计每个点被引用的次数
    point_counts = np.zeros(len(points_xyz), dtype=int)

    for indices in active_block_data.values():
        point_counts[indices] += 1

    # 2. 找出重叠点 (count > 1 的点)
    overlap_mask = point_counts > 1
    overlap_indices = np.where(overlap_mask)[0]

    print(f"Found {len(overlap_indices)} points in overlap regions.")

    if len(overlap_indices) == 0:
        print("WARNING: No overlap detected! Check expansion radius.")
        return

    # 3. 绘图
    fig, ax = plt.subplots(figsize=(20, 20))

    overlap_points = points_xyz[overlap_indices]

    # 设置视野
    view_min_x, view_min_y = np.min(overlap_points[:, :2], axis=0)
    view_max_x, view_max_y = np.max(overlap_points[:, :2], axis=0)
    margin = (view_max_x - view_min_x) * 0.1
    ax.set_xlim(view_min_x - margin, view_max_x + margin)
    ax.set_ylim(view_min_y - margin, view_max_y + margin)

    # 画背景 (淡灰)
    ax.scatter(points_xyz[::50, 0], points_xyz[::50, 1], s=1, c='lightgray', alpha=0.1)

    # 画重叠区 (高亮红色)
    ax.scatter(overlap_points[:, 0], overlap_points[:, 1], s=2, c='red', alpha=0.5, label='Overlap Region')

    ax.set_title(f"Topological Penetration Zones (Overlap Points: {len(overlap_indices)})", fontsize=20)
    ax.legend()

    vis_file = os.path.join(output_path, "partition_overlap_check.png")
    plt.savefig(vis_file, dpi=100, bbox_inches='tight')
    plt.close()

def visualize_combined2(points_xyz, active_block_data, bbox_by_new_id, output_path):
    print("Generating combined visualization (Boxes + Points)...")

    fig, ax = plt.subplots(figsize=(24, 24))
    cmap = plt.get_cmap('tab20')

    all_x, all_y, all_colors = [], [], []

    for i, (block_id, indices) in enumerate(active_block_data.items()):
        block_pts = points_xyz[indices]
        if len(block_pts) > 10000:
            block_pts = block_pts[::(len(block_pts) // 10000)]

        all_x.append(block_pts[:, 0])
        all_y.append(block_pts[:, 1])

        c = cmap(i % 20)
        all_colors.append(np.tile(c, (len(block_pts), 1)))

    if not all_x:
        return

    X = np.concatenate(all_x)
    Y = np.concatenate(all_y)
    C = np.concatenate(all_colors)

    perm = np.random.permutation(len(X))
    X, Y, C = X[perm], Y[perm], C[perm]

    view_min_x, view_min_y = np.min(X), np.min(Y)
    view_max_x, view_max_y = np.max(X), np.max(Y)
    margin_x = (view_max_x - view_min_x) * 0.1
    margin_y = (view_max_y - view_min_y) * 0.1

    ax.set_xlim(view_min_x - margin_x, view_max_x + margin_x)
    ax.set_ylim(view_min_y - margin_y, view_max_y + margin_y)

    ax.scatter(points_xyz[::100, 0], points_xyz[::100, 1],
               s=1, c='lightgray', alpha=0.2, zorder=0)

    ax.scatter(X, Y, c=C, s=3, alpha=0.7, edgecolors='none', zorder=10)

    for new_id in active_block_data.keys():
        bbox = bbox_by_new_id[new_id]
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        rect = patches.Rectangle((bbox[0], bbox[1]), width, height,
                                 linewidth=2, edgecolor='black', facecolor='none',
                                 linestyle='--', zorder=20)
        ax.add_patch(rect)

        ax.text(bbox[0] + width / 2, bbox[1] + height / 2, str(new_id),
                color='black', fontsize=16, fontweight='bold', ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.2'),
                zorder=30)

    ax.set_aspect('equal')
    ax.set_title("TECAP Partition: Hard BBox (Black) vs Soft Topology (Color)", fontsize=24)

    vis_file = os.path.join(output_path, "partition_combined.png")
    plt.savefig(vis_file, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Combined visualization saved to: {vis_file}")

def visualize_real_shapes(points_xyz, active_block_data, output_path):
    """
    可视化：直接绘制点云分布，展示 KNN 扩充后的真实不规则形状
    active_block_data: 字典 {block_id: expanded_indices}
    """
    print("Generating point-wise shape visualization...")

    fig, ax = plt.subplots(figsize=(24, 24))

    # 1. 确定视野范围 (聚焦到核心区)
    all_active_indices = np.concatenate(list(active_block_data.values()))
    active_points = points_xyz[all_active_indices]

    view_min_x, view_min_y = np.min(active_points[:, :2], axis=0)
    view_max_x, view_max_y = np.max(active_points[:, :2], axis=0)

    margin_x = (view_max_x - view_min_x) * 0.1
    margin_y = (view_max_y - view_min_y) * 0.1

    ax.set_xlim(view_min_x - margin_x, view_max_x + margin_x)
    ax.set_ylim(view_min_y - margin_y, view_max_y + margin_y)

    # 2. 绘制背景 (淡淡的灰色，表示被遗弃的噪声)
    # 为了性能，背景降采样
    ax.scatter(points_xyz[::50, 0], points_xyz[::50, 1],
               s=1, c='lightgray', alpha=0.3, label='Background Noise', zorder=0)

    # 3. 循环绘制每个 Block 的真实形状 (彩色)
    # 使用 tab20 色板，保证相邻块颜色不同
    cmap = plt.get_cmap('tab20')

    for i, (block_id, indices) in enumerate(active_block_data.items()):
        # 获取该块的所有点 (核心 + 扩充)
        block_pts = points_xyz[indices]

        # 为了可视化的清晰度，如果点太多，稍微降采样一下
        if len(block_pts) > 20000:
            step = len(block_pts) // 20000
            block_pts = block_pts[::step]

        color = cmap(i % 20)

        # 核心：Scatter Plot 画出真实形状
        ax.scatter(block_pts[:, 0], block_pts[:, 1],
                   s=3, c=[color], alpha=0.8, label=f'Block {block_id}', zorder=10 + i)

        # 标号 (标在重心位置)
        center_x = np.mean(block_pts[:, 0])
        center_y = np.mean(block_pts[:, 1])
        ax.text(center_x, center_y, str(block_id),
                color='black', fontsize=12, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5),
                zorder=100)

    ax.set_aspect('equal')
    ax.set_title(f"TECAP Real Topological Shapes (KNN Expanded)", fontsize=24)

    # 保存
    vis_file = os.path.join(output_path, "partition_real_shape.png")
    plt.savefig(vis_file, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Shape visualization saved to: {vis_file}")
def visualize_partitions2_sorted(points_xyz, bbox_by_new_id, output_path, show_orig_map=None):
    """
    按 new_id 聚焦画框：输出 partition_map_focused.png
    """
    print("Generating SORTED focused partition visualization (new_id)...")

    if not bbox_by_new_id:
        print("No training blocks to visualize!")
        return

    fig, ax = plt.subplots(figsize=(20, 20))

    bboxes = np.array(list(bbox_by_new_id.values()))
    view_min_x = np.min(bboxes[:, 0])
    view_min_y = np.min(bboxes[:, 1])
    view_max_x = np.max(bboxes[:, 2])
    view_max_y = np.max(bboxes[:, 3])

    margin_x = (view_max_x - view_min_x) * 0.1
    margin_y = (view_max_y - view_min_y) * 0.1

    ax.set_xlim(view_min_x - margin_x, view_max_x + margin_x)
    ax.set_ylim(view_min_y - margin_y, view_max_y + margin_y)

    # 背景点
    ax.scatter(points_xyz[::50, 0], points_xyz[::50, 1],
               s=1.0, c='black', alpha=0.15, zorder=0)

    # 画框
    for new_id in sorted(bbox_by_new_id.keys()):
        bbox = bbox_by_new_id[new_id]
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min

        rect = patches.Rectangle((x_min, y_min), width, height,
                                 linewidth=3, edgecolor='#00FF00', facecolor='none',
                                 alpha=1.0, zorder=20, linestyle='-')
        ax.add_patch(rect)

        label = str(new_id)
        if show_orig_map is not None and new_id in show_orig_map:
            label = f"{new_id}\n(orig {show_orig_map[new_id]})"

        ax.text(x_min + width / 2, y_min + height / 2, label,
                color='#00AA00', fontsize=14, ha='center', va='center',
                fontweight='bold', zorder=30,
                bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1))

    ax.set_aspect('equal')
    ax.set_title("TECAP Core Partition (Focused / Sorted Training Blocks / new_id)", fontsize=24)

    vis_file = os.path.join(output_path, "partition_map_focused.png")
    plt.savefig(vis_file, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Focused visualization saved to: {vis_file}")


def visualize_partitions(points_xyz, blocks, block_status, output_path):
    """
    生成分块可视化图
    points_xyz: 所有点云坐标 (N, 3)
    blocks: 分块列表
    block_status: 字典 {block_id: "status_string"} ('active', 'sparse', 'no_images')
    """
    print("Generating partition visualization...")

    # 1. 设置画布
    fig, ax = plt.subplots(figsize=(20, 20))

    # 2. 绘制背景点云 (为了速度，降采样显示，每50个点取1个)
    # 假设 Z 轴朝上，我们画 XY 平面俯视图
    downsample_rate = 50
    ax.scatter(points_xyz[::downsample_rate, 0], points_xyz[::downsample_rate, 1],
               s=0.5, c='black', alpha=0.3, label='Sparse Points')

    # 3. 绘制每个 Block 的框
    for i, block in enumerate(blocks):
        bbox = block['bbox']  # [xmin, ymin, xmax, ymax]
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min

        status = block_status.get(i, 'unknown')

        # 颜色编码
        if status == 'active':
            color = 'green'
            lw = 2
            fill = False
            zorder = 10
        elif status == 'sparse':  # 点太少
            color = 'gray'
            lw = 1
            fill = True
            alpha = 0.1
            zorder = 1
        elif status == 'no_images':  # 图片太少
            color = 'red'
            lw = 2
            fill = False
            zorder = 5
        else:
            color = 'blue'
            lw = 1
            fill = False
            zorder = 1

        # 画矩形
        rect = patches.Rectangle((x_min, y_min), width, height,
                                 linewidth=lw, edgecolor=color, facecolor=color if fill else 'none',
                                 alpha=0.3 if fill else 1.0, zorder=zorder)
        ax.add_patch(rect)

        # 标 ID
        if status == 'active' or status == 'no_images':
            ax.text(x_min + width / 2, y_min + height / 2, str(i),
                    color=color, fontsize=12, ha='center', va='center', fontweight='bold', zorder=20)

    # 4. 设置属性
    ax.set_aspect('equal')
    ax.set_title(f"TECAP Partition Result (Total Points: {len(points_xyz)})", fontsize=20)

    # 简单的图例说明
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label='Active (Training)'),
        Line2D([0], [0], color='red', lw=2, label='Skipped (Not enough images)'),
        patches.Patch(facecolor='gray', edgecolor='gray', alpha=0.3, label='Skipped (Too sparse)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=15)

    # 5. 保存
    vis_file = os.path.join(output_path, "partition_map.png")
    plt.savefig(vis_file, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {vis_file}")
# ==========================================
# 1. 辅助函数：数据加载与写入
# ==========================================

def load_colmap_scene(sparse_path):
    """自动加载 bin 或 txt 格式的 COLMAP 数据,建立id映射"""
    print(f"Loading COLMAP scene from {sparse_path}...")

    files = os.listdir(sparse_path)
    use_binary = "cameras.bin" in files and "images.bin" in files and "points3D.bin" in files

    if use_binary:
        print("Detected BINARY format.")
        cameras = colmap_utils.read_intrinsics_binary(os.path.join(sparse_path, "cameras.bin"))
        images = colmap_utils.read_extrinsics_binary(os.path.join(sparse_path, "images.bin"))
        points3D_xyz, points3D_rgb, points3D_err = colmap_utils.read_points3D_binary(
            os.path.join(sparse_path, "points3D.bin"))
        # 注意：你的 loader 返回的是 (xyz, rgb, err) 的 tuple，不是 dict
        # 为了后续方便，我们需要把 image_ids 关联起来。
        # 但你的 read_points3D_binary 实现似乎丢弃了 image_ids 和 point2D_idxs (只读了前几个属性)
        # 如果必须用 track 信息，我们需要修改 loader 或者重新读取。
        # 鉴于你的 loader 代码里 read_points3D_binary 确实跳过了 track_elems，
        # 我们这里暂时只能用 images 里的 point3D_ids 反查。
        # 重新读取 ID
        point_ids = []
        with open(os.path.join(sparse_path, "points3D.bin"), "rb") as fid:
            num_points = struct.unpack("<Q", fid.read(8))[0]
            for _ in range(num_points):
                # 格式: ID(8), XYZ(24), RGB(3), ERR(8), TrackLen(8), Tracks(...)
                props = struct.unpack("<QdddBBBd", fid.read(43))
                point_id = props[0]
                point_ids.append(point_id)
                track_len = struct.unpack("<Q", fid.read(8))[0]
                fid.seek(8 * track_len, 1)  # 跳过 tracks

        point_ids = np.array(point_ids)
    else:
        print("Detected TEXT format.")
        cameras = colmap_utils.read_intrinsics_text(os.path.join(sparse_path, "cameras.txt"))
        images = colmap_utils.read_extrinsics_text(os.path.join(sparse_path, "images.txt"))
        points3D_xyz, points3D_rgb, points3D_err = colmap_utils.read_points3D_text(
            os.path.join(sparse_path, "points3D.txt"))
        point_ids = []
        with open(os.path.join(sparse_path, "points3D.txt"), "r") as fid:
            while True:
                line = fid.readline()
                if not line: break
                if line.startswith("#"): continue
                point_ids.append(int(line.split()[0]))
        point_ids = np.array(point_ids)
    print(f"  Verified: Loaded {len(point_ids)} Point IDs.")
    # 重新构建 Points 字典以便索引 (ID -> XYZ)
    # 注意：你的 loader 返回的是数组，ID 信息丢失了（假设是连续的？）
    # 通常 read_points3D_binary 会返回字典 {id: Point3D}。
    # 如果你的 loader 返回的是 numpy array，那我们假设 index+1 就是 ID 或者需要修改 loader。
    # 为了稳健，这里我们假设 point3D_xyz 的索引对应 images 里的 point3D_ids。

    # 修正：为了配合你的 loader 输出 (numpy array)，我们创建一个简单的 wrapper
    # points_dict = {}
    # for idx, xyz in enumerate(points3D_xyz):
    #     # 假设 loader 是按顺序读的，但真实 ID 可能不连续。
    #     # 这是一个潜在风险点。建议检查 loader 是否能返回 ID。
    #     # 临时方案：我们用 idx 作为 ID，但这可能导致与 images.txt 里的 id 对不上。
    #     # **强烈建议**：使用 standard colmap loader 或修改你的 loader 返回 ID。
    #     # 这里为了演示 TECAP 逻辑，我们先假设 image_ids 里的点 ID 是准确的，
    #     # 并且我们只关心空间位置划分。
    #     points_dict[idx] = xyz
    # 2. 建立映射：真实ID -> 数组下标 (用于筛选图片)
    id_to_idx = {pid: i for i, pid in enumerate(point_ids)}

    # 3. 建立映射：数组下标 -> 真实ID (用于保存子场景)
    idx_to_id = {i: pid for i, pid in enumerate(point_ids)}

    return cameras, images, points3D_xyz, points3D_rgb,id_to_idx,idx_to_id


def estimate_normals(points_xyz):
    """使用 Open3D 计算点云法线 (QGS 需要)"""
    print("Estimating Normals via Open3D...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))
    return np.asarray(pcd.normals)


def save_colmap_subset(output_folder, cameras, subset_images, subset_points_indices, all_points_xyz, all_points_rgb):
    """将筛选后的数据保存为 COLMAP text 格式"""
    os.makedirs(output_folder, exist_ok=True)

    # 1. Copy Cameras (假设全用)
    # 为了简化，直接把原始 cameras.txt/bin 转写或复制
    # 这里写一个简易的 cameras.txt writer
    with open(os.path.join(output_folder, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera.\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for cam in cameras.values():
            params_str = " ".join(map(str, cam.params))
            f.write(f"{cam.id} {cam.model} {cam.width} {cam.height} {params_str}\n")

    # 2. Write Images
    # 只保留 subset_images 里的图
    # 并且把看不到的点（不在 subset_points_indices 里）设为 -1 (虽然 QGS 可能不care track)
    valid_point_set = set(subset_points_indices)
    with open(os.path.join(output_folder, "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image.\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

        for img_id in subset_images:
            img = subset_images[img_id]
            q = img.qvec
            t = img.tvec
            f.write(f"{img.id} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {img.camera_id} {img.name}\n")

            # Filter points2D
            p2d_lines = []
            for xy, p3d_id in zip(img.xys, img.point3D_ids):
                # 如果这个点不在我们切分的块里，就把它标记为 -1 (未观测)
                # 这样训练时就不会用到这些点
                # 注意：这里有一个 ID 映射问题，取决于你的 loader 返回的 ID
                final_id = p3d_id if p3d_id in valid_point_set else -1
                p2d_lines.append(f"{xy[0]} {xy[1]} {final_id}")

            f.write(" ".join(p2d_lines) + "\n")

    # 3. Write Points3D
    with open(os.path.join(output_folder, "points3D.txt"), "w") as f:
        f.write("# 3D point list with one line of data per point.\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")

        for idx in subset_points_indices:
            xyz = all_points_xyz[idx]
            rgb = all_points_rgb[idx]
            # ID, X, Y, Z, R, G, B, ERR, TRACKS...
            # Error 和 Tracks 暂时留空或写 dummy，因为 QGS 初始化主要只看 XYZ RGB
            f.write(f"{idx} {xyz[0]} {xyz[1]} {xyz[2]} {int(rgb[0])} {int(rgb[1])} {int(rgb[2])} 0.0\n")


# ==========================================
# 2. 核心逻辑类 (TECAP Strategy)
# ==========================================

class TopologyPartitioner:
    def __init__(self, points, max_points_per_block=1500_00):  # 调小点数以便测试
        self.points = points
        self.max_points = max_points_per_block
        self.blocks = []
        self.global_tree = cKDTree(points)

    def build_quadtree(self, indices, bbox):
        if len(indices) <= self.max_points:
            self.blocks.append({"core_indices": indices, "bbox": bbox})
            return

        x_min, y_min, x_max, y_max = bbox
        mid_x, mid_y = (x_min + x_max) / 2, (y_min + y_max) / 2

        masks = [
            (self.points[indices, 0] <= mid_x) & (self.points[indices, 1] <= mid_y),
            (self.points[indices, 0] > mid_x) & (self.points[indices, 1] <= mid_y),
            (self.points[indices, 0] <= mid_x) & (self.points[indices, 1] > mid_y),
            (self.points[indices, 0] > mid_x) & (self.points[indices, 1] > mid_y)
        ]
        bboxes = [
            [x_min, y_min, mid_x, mid_y], [mid_x, y_min, x_max, mid_y],
            [x_min, mid_y, mid_x, y_max], [mid_x, mid_y, x_max, y_max]
        ]

        for i in range(4):
            if np.any(masks[i]):
                self.build_quadtree(indices[masks[i]], bboxes[i])

    def expand_block_topology(self, block_data, k_neighbors=50):
        # KNN 拓扑扩展：哪怕被切断，也要把邻居抓进来
        core_indices = block_data['core_indices']
        if len(core_indices) == 0: return []

        # 为了速度，只对核心区采样部分点做 query，或者全部做
        # 这里全部做，保证边界完整
        dists, neighbors_indices = self.global_tree.query(self.points[core_indices], k=k_neighbors)
        return np.unique(neighbors_indices.flatten())

    def expand_block_surface_crawling(self, block_data, normals, max_depth=5, similarity_thresh=0.9):
        """
        【TECAP V2 核心创新】：法线引导的表面爬取
        不再是简单的空间扩充，而是顺着几何表面（墙面/地面）向外生长。

        block_data: 当前区块信息
        normals: 全局点云法线 (N, 3)
        max_depth: 爬取迭代次数 (次数越多，爬得越远)
        similarity_thresh: 法线相似度阈值 (cos theta), 0.9 大约是 25度以内
        """
        core_indices = block_data['core_indices']
        if len(core_indices) == 0: return []

        current_indices = core_indices
        all_indices = set(core_indices)

        # 1. 确定边界点 (为了加速，只从边缘开始爬)
        # 简单判定：离 bbox 边界最近的那部分点
        # 这里为了代码简洁，我们直接用所有点做第一轮查询，后续只用新点做查询
        # 工程优化：实际应该先算个凸包或边缘检测，这里简化处理
        frontier = current_indices

        print(f"    -> Surface Crawling (Depth {max_depth}): ", end="")

        for i in range(max_depth):
            # 1. 搜索邻居 (K=20 足够了，我们要的是连通性)
            # 随着迭代，frontier 会变成新加入的点
            if len(frontier) == 0: break

            # 降采样 frontier 以加速 (如果边缘点太多)
            query_points = self.points[frontier]
            if len(query_points) > 5000:
                query_points = query_points[::(len(query_points) // 5000)]

            dists, neighbors = self.global_tree.query(query_points, k=20)

            # 展平
            neighbors = np.unique(neighbors.flatten())

            # 2. 【核心】法线一致性检查 (Normal Consistency Check)
            # 我们需要比较 neighbors 的法线 和 它们来源点 的法线
            # 这里做一个近似：如果邻居的法线，和该 Block 的"平均法线"或"最近点法线"一致
            # 更精确的做法是点对点比较，但太慢。
            # 这里采用"聚类一致性"：
            # 如果 neighbor 的法线 与 扩展前沿(frontier) 中任意点的法线相似，就保留。

            # 向量化计算：
            # 获取所有候选邻居的法线
            cand_normals = normals[neighbors]  # (M, 3)

            # 获取前沿点的平均法线方向 (这有点粗糙，但能保证大方向一致)
            # 更好的方法：只保留那些法线与"当前生长方向"一致的点
            # 为了实现"顺着墙爬"，我们只通过 KDTree 的距离关系隐式关联，这里只过滤噪声

            # 实际上，只要是同一个物体，空间距离近+法线存在，通常就是连通的。
            # 我们可以只过滤掉法线差异极大的点 (比如拐角后的点)

            # 既然我们无法快速做点对点法线对比，我们采用一种"自适应过滤"：
            # 只有当新点不在 current_set 中，且距离够近时才加入

            # 找出尚未加入的点
            new_candidates = np.setdiff1d(neighbors, list(all_indices))
            if len(new_candidates) == 0: break

            # --- 法线过滤 ---
            # 我们随机取一些 frontier 的法线作为参考
            ref_normals = normals[np.random.choice(frontier, min(100, len(frontier)))]
            cand_normals = normals[new_candidates]

            # 计算相似度矩阵 (M_cand, N_ref) -> 取最大相似度
            # 如果一个候选点，和任意一个前沿点的法线相似，它就是墙的一部分
            dot_products = np.dot(cand_normals, ref_normals.T)  # (M, 100)
            max_sim = np.max(dot_products, axis=1)  # (M,)

            # 只有相似度 > 0.9 的才保留
            valid_mask = max_sim > similarity_thresh
            valid_new_points = new_candidates[valid_mask]

            if len(valid_new_points) == 0: break

            # 更新状态
            all_indices.update(valid_new_points)
            frontier = valid_new_points  # 下一轮只从新长出来的点开始爬

            print(f".", end="")  # 进度条

        print(f" Done. (+{len(all_indices) - len(core_indices)} pts)")
        return np.array(list(all_indices))

class CurvatureViewSelector:
    def __init__(self, images,id_to_idx_map):
        self.images = images
        self.id_map = id_to_idx_map

    def select_best_images(self, block_point_indices, all_points, all_normals, top_k_images=1000):
        """
        TECAP Step 2: 结合 距离 + 法线夹角 + 覆盖率 筛选图片
        """
        if len(block_point_indices) == 0: return {}

        # 优化：建立临时反向索引
        block_indices_set = set(block_point_indices)
        relevant_images_score = {}  # img_id -> score



        min_overlap = 50

        print(f"  Evaluating images (Min overlap required: {min_overlap} points)...")

        for img_id, img in self.images.items():
            # img.point3D_ids 存的是“真实ID”
            # 我们需要把它们转成“数组下标”再去查集合

            valid_indices = []
            for real_id in img.point3D_ids:
                if real_id != -1 and real_id in self.id_map:
                    idx = self.id_map[real_id]
                    if idx in block_indices_set:
                        valid_indices.append(idx)

            intersect_count = len(valid_indices)

            if intersect_count < min_overlap:
                continue

            # 3. 计算分数
            # 3.1 距离分数 (相机中心到这些点的平均距离)
            cam_center = -np.matmul(img.qvec2rotmat().T, img.tvec)
            points_xyz = all_points[valid_indices]
            dists = np.linalg.norm(points_xyz - cam_center, axis=1)
            avg_dist = np.mean(dists)
            score_dist = 1.0 / (avg_dist + 1e-5)

            # 3.2 法线分数 (视线与法线的平均夹角)
            points_normal = all_normals[valid_indices]
            view_vecs = points_xyz - cam_center
            # 归一化视线向量
            view_vecs /= (np.linalg.norm(view_vecs, axis=1, keepdims=True) + 1e-10)

            # 点积计算夹角:
            # 理想情况是视线(view_vec)指向点，法线(normal)指向外，两者夹角180度，点积为-1
            # 或者视线是从点指向相机，两者夹角0度。
            # 这里统一取绝对值衡量“平行程度”(即视线是否垂直于表面)
            dots = np.sum(view_vecs * points_normal, axis=1)
            score_angle = np.mean(np.abs(dots))

            # 3.3 综合评分 (距离权重 0.5，角度权重 0.5)
            final_score = 0.5 * score_dist + 0.5 * score_angle
            relevant_images_score[img_id] = final_score

        # 4. 排序与智能截断
        if not relevant_images_score:
            return {}

        sorted_imgs = sorted(relevant_images_score.items(), key=lambda x: x[1], reverse=True)

        # 【过滤策略2】：相对分数筛选
        # 获取最高分作为基准
        best_score = sorted_imgs[0][1]

        final_selection = []
        for i, (img_id, score) in enumerate(sorted_imgs):
            # 逻辑：
            # A. 前 20 张图无论分数如何，强制保留 (保底覆盖)
            # B. 20 张之后的图，如果分数低于最高分的 50%，则丢弃 (去粗取精)
            # C. 总数量不超过 top_k_images

            if len(final_selection) >= top_k_images:
                break

            if i < 20:
                final_selection.append(img_id)
            elif score > best_score * 0.5:
                final_selection.append(img_id)
            else:
                # 因为是排好序的，一旦低于阈值，后面肯定更低，直接退出
                break

                # 返回选中的 Image 对象字典
        return {k: self.images[k] for k in final_selection}



# ==========================================
# 3. 主程序入口
# ==========================================

def run_tecap_pipeline(source_path, output_path, python_path="python"):
    sparse_path = os.path.join(source_path, "sparse", "0")
    if not os.path.exists(sparse_path):
        print(f"Error: Sparse folder not found at {sparse_path}")
        return

    # 1. 加载数据
    cameras, images, points_xyz, points_rgb,id_to_idx,idx_to_id = load_colmap_scene(sparse_path)
    print(f"Loaded {len(points_xyz)} points, {len(images)} images.")

    # 2. 计算法线 (TECAP 关键依赖)
    points_normals = estimate_normals(points_xyz)

    # 3. 拓扑划分 (TECAP Step 1)
    print("Partitioning Space...")
    partitioner = TopologyPartitioner(points_xyz)
    root_bbox = [np.min(points_xyz[:, 0]), np.min(points_xyz[:, 1]),
                 np.max(points_xyz[:, 0]), np.max(points_xyz[:, 1])]
    partitioner.build_quadtree(np.arange(len(points_xyz)), root_bbox)
    print(f"Divided scene into {len(partitioner.blocks)} blocks.")

    block_status={}
    # active_block_data={}

    # 4. 准备分块数据
    view_selector = CurvatureViewSelector(images,id_to_idx)
    valid_blocks_metadata=[]

    for i, block in enumerate(partitioner.blocks):
        # 【过滤器 1】：点数太少，直接跳过（解决 Block 0,1,2 刷屏问题）
        if len(block['core_indices']) < 6000:
            # print(f"Skipping Block {i} (Too sparse)...") # 嫌吵可以注释掉
            block_status[i]='sparse'
            continue
        print(f"\n=== Preparing Block {i} ===")

        # A. 拓扑扩充 (KNN)
        # train_indices = partitioner.expand_block_topology(block)

        # A. 拓扑扩充 (使用 Surface Crawling)
        # max_depth=5 意味着它会顺着墙面“爬” 5次邻域搜索的距离（大概几米）
        # 并且只会爬平面，不会爬过直角拐角
        train_indices = partitioner.expand_block_surface_crawling(
            block, points_normals, max_depth=6, similarity_thresh=0.85
        )

        print(f"  Core points: {len(block['core_indices'])}, Expanded: {len(train_indices)}")

        # B. 视角选择 (曲率敏感)
        subset_images = view_selector.select_best_images(train_indices, points_xyz, points_normals)
        print(f"  Selected {len(subset_images)} images (from {len(images)}).")
        # 【过滤器 2】：图片太少，不训练（解决 Block 9, 25, 30 这种边缘块）
        img_count=len(subset_images)
        if len(subset_images) < 20:
            print(f"  Skipping Block {i}: Only {len(subset_images)} valid images (Need 20+).")
            block_status[i]='no_images'
            continue

        block_status[i]='active'

        valid_blocks_metadata.append({
            "original_index": i,
            "bbox": block['bbox'],
            "train_indices": train_indices,
            "images": subset_images,
            "image_count": img_count,
            "core_points": len(block['core_indices']),
            "expanded_points": len(train_indices)
        })

        # 按图片数量降序，图片一样则按扩充后点数降序
        valid_blocks_metadata.sort(key=lambda x: (x['image_count'], x['expanded_points']), reverse=True)
        active_block_data = {}  # new_id -> expanded_indices
        bbox_by_new_id = {}  # new_id -> bbox
        new_id_to_original = {}  # new_id -> original_index（可选，便于调试）

        sorted_blocks_list=[]
        for new_id, meta in enumerate(valid_blocks_metadata):
            orig=meta["original_index"]
            subset_images = meta['images']
            train_indices = meta['train_indices']
            bbox = meta['bbox']

            # 打印信息：展示 原始ID -> 新ID 的映射
            print(
                f"  [Block {new_id}] (was Original {meta['original_index']}): {meta['image_count']} images, {meta['expanded_points']} points (Core: {meta['core_points']}).")

            # 收集可视化数据
            active_block_data[new_id] = train_indices
            bbox_by_new_id[new_id] = bbox
            new_id_to_original[new_id] = orig
            sorted_blocks_list.append({'bbox': bbox})

            # C. 保存子场景 (文件夹名为 block_0, block_1...)
            block_name = f"block_{new_id}"
            block_root = os.path.join(output_path, block_name)
            save_path = os.path.join(block_root, "sparse", "0")

            if not os.path.exists(save_path):
                save_colmap_subset(save_path, cameras, subset_images, train_indices, points_xyz, points_rgb)

            # D. 生成训练命令
            qgs_output = os.path.join(output_path, "trained", block_name)
            os.makedirs(qgs_output, exist_ok=True)

            # 保存原始 BBox (用于 Fusion 时的幽灵裁剪)
            np.savetxt(os.path.join(qgs_output, "bbox.txt"), bbox)

            cmd = [
                python_path, "train.py",
                "-s", block_root,
                "-m", qgs_output,
                "--iterations", "30000"
            ]

            # 取消注释以运行
            # subprocess.run(cmd, check=True)
            # print(f"     Command: {' '.join(cmd)}")
    visualize_partitions2_sorted(points_xyz, bbox_by_new_id, output_path,show_orig_map=new_id_to_original)
    visualize_real_shapes(points_xyz,active_block_data,output_path)
    visualize_combined2(points_xyz,active_block_data,bbox_by_new_id,output_path)
    visualize_overlaps_only(points_xyz,active_block_data,output_path)
    print("\nAll blocks processed. Ready for TSDF Fusion.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Path to source folder containing sparse/0/")
    parser.add_argument("--output", type=str, required=True, help="Path to save partitioned blocks")
    args = parser.parse_args()

    run_tecap_pipeline(args.source, args.output)