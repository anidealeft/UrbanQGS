import os
import numpy as np
import collections
import struct
import argparse
from scipy.spatial import cKDTree
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import colmap_loader as colmap_utils

# ==========================================
# 1. 核心分块器：基于相机负载均衡与空域感知
# ==========================================

class VALBPartitioner:
    def __init__(self, cameras, images, points_xyz, id_to_idx,idx_to_id):
        self.cameras = cameras
        self.images = images
        self.points_xyz = points_xyz
        self.id_to_idx = id_to_idx
        self.idx_to_id = idx_to_id
        self.global_tree = cKDTree(points_xyz)

        # 预计算所有相机的中心点
        self.img_ids = list(self.images.keys())
        self.cam_centers = np.array([self._get_cam_center(self.images[i]) for i in self.img_ids])

    def _get_cam_center(self, img):
        R = img.qvec2rotmat()
        t = img.tvec
        return -np.matmul(R.T, t)

    def partition_balanced(self, m_splits=3, n_splits=3):
        """
        核心改进：基于地理空间等分网格，确保 core_bbox 不重叠且图不乱
        """
        print(f"Partitioning scene into {m_splits}x{n_splits} geometric grids...")

        # 1. 使用分位数剔除离群点 (只取中间 98% 的点来定边界)
        # 这样可以确保网格紧贴实际的建筑和相机区域
        xmin = np.percentile(self.points_xyz[:, 0], 1)
        xmax = np.percentile(self.points_xyz[:, 0], 99)
        ymin = np.percentile(self.points_xyz[:, 1], 1)
        ymax = np.percentile(self.points_xyz[:, 1], 99)

        w_step = (xmax - xmin) / m_splits
        h_step = (ymax - ymin) / n_splits

        blocks_metadata = []
        for i in range(m_splits):
            for j in range(n_splits):
                # 计算该块的硬地理边界 (Core BBox)
                b_xmin = xmin + i * w_step
                b_xmax = xmin + (i + 1) * w_step
                b_ymin = ymin + j * h_step
                b_ymax = ymin + (j + 1) * h_step

                core_bbox = [b_xmin, b_ymin, b_xmax, b_ymax]

                # 2. 找到中心落在该区域内的“基础相机”
                base_img_ids = []
                base_cam_centers = []
                for idx, img_id in enumerate(self.img_ids):
                    center = self.cam_centers[idx]
                    if (b_xmin <= center[0] < b_xmax) and (b_ymin <= center[1] < b_ymax):
                        base_img_ids.append(img_id)
                        base_cam_centers.append(center)

                if not base_img_ids:
                    # 如果该区域没有相机中心，尝试放宽要求，或者记录为空
                    print(f"  Warning: Block {len(blocks_metadata)} has no camera centers.")

                blocks_metadata.append({
                    "core_bbox": core_bbox,
                    "base_img_ids": base_img_ids,
                    "cam_centers": np.array(base_cam_centers) if base_cam_centers else np.empty((0, 3))
                })
        return blocks_metadata

    def expand_visibility(self, block, padding_ratio=0.1, vis_threshold=0.2):
        """
        改进：严格矩形裁剪点云 + 高阈值视角筛选
        """
        xmin, ymin, xmax, ymax = block['core_bbox']
        w, h = xmax - xmin, ymax - ymin

        # 1. 严格矩形裁剪点云：只拿 Core + Padding 范围内的点
        p_indices = np.where(
            (self.points_xyz[:, 0] >= xmin - w * padding_ratio) &
            (self.points_xyz[:, 0] <= xmax + w * padding_ratio) &
            (self.points_xyz[:, 1] >= ymin - h * padding_ratio) &
            (self.points_xyz[:, 1] <= ymax + h * padding_ratio)
        )[0]

        # 2. 核心区点 ID 集合 (用于可见性判断)
        core_p_indices = np.where(
            (self.points_xyz[:, 0] >= xmin) & (self.points_xyz[:, 0] <= xmax) &
            (self.points_xyz[:, 1] >= ymin) & (self.points_xyz[:, 1] <= ymax)
        )[0]
        core_point_ids_set = set([self.idx_to_id[i] for i in core_p_indices])

        # 3. 筛选可见相机
        final_img_ids = set(block['base_img_ids'])

        # 遍历所有 713 张图
        for iid, img in self.images.items():
            if iid in final_img_ids: continue

            visible_pids = [pid for pid in img.point3D_ids if pid != -1]
            if not visible_pids: continue

            # 计算该相机看到的点中有多少落在当前 Block 的地理核心区
            overlap_count = sum(1 for pid in visible_pids if pid in core_point_ids_set)
            overlap_ratio = overlap_count / len(visible_pids)

            # 阈值设为 0.5，意味着只有一半视野在该块内的外部相机才会被引入
            if overlap_ratio > vis_threshold:
                final_img_ids.add(iid)

        return list(p_indices), list(final_img_ids)
# ==========================================
# 2. 数据加载与保存工具
# ==========================================

def load_scene(sparse_path):
    print(f"Loading COLMAP data...")
    files = os.listdir(sparse_path)
    is_bin = "images.bin" in files
    
    if is_bin:
        cams = colmap_utils.read_intrinsics_binary(os.path.join(sparse_path, "cameras.bin"))
        imgs = colmap_utils.read_extrinsics_binary(os.path.join(sparse_path, "images.bin"))
        pts_xyz, pts_rgb, _ = colmap_utils.read_points3D_binary(os.path.join(sparse_path, "points3D.bin"))
        # 获取原始 ID
        with open(os.path.join(sparse_path, "points3D.bin"), "rb") as f:
            num = struct.unpack("<Q", f.read(8))[0]
            pids = [struct.unpack("<QdddBBBd", f.read(43))[0] for _ in range(num)]
            # 简单跳过 track 处理加速加载
    else:
        cams = colmap_utils.read_intrinsics_text(os.path.join(sparse_path, "cameras.txt"))
        imgs = colmap_utils.read_extrinsics_text(os.path.join(sparse_path, "images.txt"))
        pts_xyz, pts_rgb, _ = colmap_utils.read_points3D_text(os.path.join(sparse_path, "points3D.txt"))
        with open(os.path.join(sparse_path, "points3D.txt"), "r") as f:
            pids = [int(line.split()[0]) for line in f if not line.startswith("#")]

    id_to_idx = {pid: i for i, pid in enumerate(pids)}
    idx_to_id={i: pid for i, pid in enumerate(pids)}
    return cams, imgs, pts_xyz, pts_rgb, id_to_idx,idx_to_id

def save_subset(path, cameras, images_dict, p_indices, all_xyz, all_rgb):
    os.makedirs(os.path.join(path, "sparse/0"), exist_ok=True)
    out = os.path.join(path, "sparse/0")
    
    # 写入 cameras.txt
    with open(os.path.join(out, "cameras.txt"), "w") as f:
        for c in cameras.values():
            f.write(f"{c.id} {c.model} {c.width} {c.height} {' '.join(map(str, c.params))}\n")
            
    # 写入 images.txt
    valid_p_set = set(p_indices)
    with open(os.path.join(out, "images.txt"), "w") as f:
        for iid in images_dict:
            img = images_dict[iid]
            q, t = img.qvec, img.tvec
            f.write(f"{img.id} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {img.camera_id} {img.name}\n")
            p2d = [f"{xy[0]} {xy[1]} {pid if pid in valid_p_set else -1}" 
                   for xy, pid in zip(img.xys, img.point3D_ids)]
            f.write(" ".join(p2d) + "\n")

    # 写入 points3D.txt
    with open(os.path.join(out, "points3D.txt"), "w") as f:
        for idx in p_indices:
            xyz, rgb = all_xyz[idx], all_rgb[idx]
            f.write(f"{idx} {xyz[0]} {xyz[1]} {xyz[2]} {int(rgb[0])} {int(rgb[1])} {int(rgb[2])} 0.0\n")

# ==========================================
# 3. 主流程
# ==========================================

def run_partition(args):
    sparse_path = os.path.join(args.source, "sparse/0")
    cams, imgs, xyz, rgb, id_to_idx ,idx_to_id= load_scene(sparse_path)
    
    partitioner = VALBPartitioner(cams, imgs, xyz, id_to_idx,idx_to_id)
    # 按照 3x3 划分为 9 个平衡块
    blocks = partitioner.partition_balanced(m_splits=3, n_splits=3)
    
    for i, block in enumerate(blocks):
        print(f"Processing Block {i}...")
        p_idx, subset_img_ids = partitioner.expand_visibility(block)
        subset_imgs = {iid: imgs[iid] for iid in subset_img_ids}
        
        block_dir = os.path.join(args.output, f"block_{i}")
        save_subset(block_dir, cams, subset_imgs, p_idx, xyz, rgb)
        
        # 保存用于后期合并裁剪的 BBox
        np.savetxt(os.path.join(block_dir, "bbox.txt"), block['core_bbox'])
        print(f"  Block {i} saved: {len(subset_img_ids)} images, {len(p_idx)} points.")

    print("\nPartitioning Done. Sub-scenes created in:", args.output)
    print("You can now run 'python train.py -s <block_path>' for each block.")


def visualize_valb_result(points_xyz, blocks, output_path):
    """
    升级版可视化：背景点云 + 核心区点云 + 彩色相机簇 + BBox
    """
    print("Generating Detailed VALB Partitioning visualization...")
    fig, ax = plt.subplots(figsize=(24, 24))

    # 1. 绘制全场景背景点云 (极低密度，仅作轮廓参考)
    ax.scatter(points_xyz[::200, 0], points_xyz[::200, 1], s=0.1, c='gray', alpha=0.2, zorder=0)

    cmap = plt.get_cmap('tab20')

    for i, block in enumerate(blocks):
        color = cmap(i % 20)
        bbox = block['core_bbox']

        # 2. 绘制属于该块的相机中心 (彩色点)
        if 'cam_centers' in block and len(block['cam_centers']) > 0:
            cams = block['cam_centers']
            ax.scatter(cams[:, 0], cams[:, 1], s=15, color=color, marker='^', label=f'Block {i} Cams', zorder=5)

        # 3. 绘制该块核心区内的真实点云 (重要：让你看清点云分布)
        # 临时筛选该 bbox 内的点进行显示
        xmin, ymin, xmax, ymax = bbox
        mask = (points_xyz[::50, 0] >= xmin) & (points_xyz[::50, 0] <= xmax) & \
               (points_xyz[::50, 1] >= ymin) & (points_xyz[::50, 1] <= ymax)
        block_pts = points_xyz[::50][mask]

        if len(block_pts) > 0:
            ax.scatter(block_pts[:, 0], block_pts[:, 1], s=1, color=color, alpha=0.5, zorder=2)

        # 4. 绘制核心 BBox 虚线框
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height,
                                 linewidth=2, edgecolor=color, facecolor='none',
                                 linestyle='--', alpha=0.9, zorder=10)
        ax.add_patch(rect)

        # 5. 标注 Block ID 和统计信息
        ax.text(xmin + width / 2, ymin + height / 2, f"ID: {i}\nCams: {len(block['base_img_ids'])}",
                color='black', fontsize=14, fontweight='bold', ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor=color, boxstyle='round,pad=0.3'),
                zorder=15)

    ax.set_aspect('equal')
    ax.set_title("VALB Partitioning: Points (Dots) & Camera Centers (Triangles)", fontsize=24)

    # 自动调整视角到点云有效区域 (剔除离群点干扰)
    x_low = np.percentile(points_xyz[:, 0], 0.5)
    x_high = np.percentile(points_xyz[:, 0], 99.5)
    y_low = np.percentile(points_xyz[:, 1], 0.5)
    y_high = np.percentile(points_xyz[:, 1], 99.5)
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_low, y_high)

    vis_file = os.path.join(output_path, "valb_partition_map_detailed.png")
    plt.savefig(vis_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Detailed visualization saved to: {vis_file}")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    # 加载数据 (假设你已有 colmap_loader)
    sparse_path = os.path.join(args.source, "sparse/0")
    cams, imgs, xyz, rgb, id_to_idx ,idx_to_id= load_colmap_scene(sparse_path)  # 替换为你现有的加载逻辑

    # 初始化分块器
    partitioner = VALBPartitioner(cams, imgs, xyz, id_to_idx,idx_to_id)

    # 执行 3x3 均衡切分 (共 9 块)
    blocks = partitioner.partition_balanced(m_splits=3, n_splits=3)

    # 调用可视化
    os.makedirs(args.output, exist_ok=True)
    visualize_valb_result(xyz, blocks, args.output)
    run_partition(parser.parse_args())