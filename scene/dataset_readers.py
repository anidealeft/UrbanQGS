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

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model2 import BasicPointCloud
import open3d as o3d

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image_path: str
    image_name: str
    image_type: str
    width: int
    height: int
    downsample: float
    cam_intr: np.array
    use_alpha: bool

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, dataset):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            params = np.array([intr.params[0],intr.params[0],intr.params[1],intr.params[2]],dtype=np.float32)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            params = np.array([intr.params[0],intr.params[1],intr.params[2],intr.params[3]],dtype=np.float32)
        elif intr.model=="OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            params = np.array([intr.params[0],intr.params[1],intr.params[2],intr.params[3]],dtype=np.float32)
        elif intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            params = np.array([intr.params[0],intr.params[0],intr.params[1],intr.params[2]],dtype=np.float32)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, extr.name)
        image_name = os.path.basename(image_path).split(".")[0]
        
        if not os.path.exists(image_path) or "sky_mask" in image_path:
            print("skip =====", image_path)
            continue

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, downsample=dataset.downsample,
                              image_path=image_path, image_name=image_name, 
                              width=width, height=height, cam_intr = params, use_alpha = dataset.use_alpha,image_type=dataset.type)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    # normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    
    return BasicPointCloud(points=positions, colors=colors, normals=None)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images_dir, dataset):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=images_dir, dataset=dataset)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if dataset.eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % dataset.hold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % dataset.hold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info
def readBlenderExtrinsicsCameras(
        colmap_root,
        images_dir,
        dataset,
        blender_extrinsics_json,
        force_resolution=None,
        mesh_transform_json=None,
        blender_intrinsics_json=None,
):
    """
    Read blender extrinsics JSON and convert to CameraInfo list usable by SPCG-QGS.

    If blender_intrinsics_json is provided:
        Use intrinsics from Blender (fx,fy,cx,cy,width,height) in that file.
    Else:
        Use intrinsics from COLMAP sparse/0/cameras.(bin|txt) (legacy behavior).

    Extrinsics: blender_extrinsics_json contains "cameras":[{"world_to_camera":4x4, ...}, ...]
    Alignment: if mesh_transform_json is provided, apply w2c_qgs = w2c_bl @ M (M is mesh_to_world).
    Coord conversion: Blender cam coords -> COLMAP/OpenCV cam coords via flip Y,Z.
    """

    # ---------------- 0) load Blender extrinsics ----------------
    with open(blender_extrinsics_json, "r", encoding="utf-8") as f:
        bex = json.load(f)
    cameras = bex.get("cameras", None)
    if cameras is None or len(cameras) == 0:
        raise ValueError(f"[BlenderExtrinsics] No 'cameras' found in {blender_extrinsics_json}")

    # ---------------- 1) choose intrinsics source ----------------
    if blender_intrinsics_json is not None:
        # ======= Use Blender intrinsics (your ORIGINAL camera) =======
        with open(blender_intrinsics_json, "r", encoding="utf-8") as f:
            intr = json.load(f)

        width = int(intr["width"])
        height = int(intr["height"])
        fx, fy, cx, cy = map(float, intr["fxfy_cxcy"])

        # optional override output resolution: scale K accordingly
        if force_resolution is not None:
            wt, ht = int(force_resolution[0]), int(force_resolution[1])
            sx = wt / width
            sy = ht / height
            fx *= sx
            fy *= sy
            cx *= sx
            cy *= sy
            width, height = wt, ht

        # FoV from K (most robust)
        FovX = 2.0 * np.arctan(width / (2.0 * fx))
        FovY = 2.0 * np.arctan(height / (2.0 * fy))
        cam_intr = np.array([fx, fy, cx, cy], dtype=np.float32)

        print("[Intrinsics] Using Blender intrinsics from:", blender_intrinsics_json)
        print(f"[K] fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}, res={width}x{height}")

        model_tag = "BLENDER_PINHOLE"

    else:
        # ======= Use COLMAP intrinsics (legacy) =======
        try:
            cameras_intrinsic_file = os.path.join(colmap_root, "sparse/0", "cameras.bin")
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_intrinsic_file = os.path.join(colmap_root, "sparse/0", "cameras.txt")
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

        cam_id = list(cam_intrinsics.keys())[0]
        intr = cam_intrinsics[cam_id]
        model = intr.model
        p = intr.params

        width = int(intr.width)
        height = int(intr.height)

        if model == "SIMPLE_PINHOLE":
            f, cx, cy = p[0], p[1], p[2]
            fx, fy = f, f
        elif model == "PINHOLE":
            fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        elif model == "OPENCV":
            fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        elif model == "SIMPLE_RADIAL":
            f, cx, cy = p[0], p[1], p[2]
            fx, fy = f, f
        else:
            raise AssertionError(f"[BlenderExtrinsics] Unsupported COLMAP camera model: {model}")

        # If you want to render at a different resolution than COLMAP intrinsics resolution:
        if force_resolution is not None:
            wt, ht = int(force_resolution[0]), int(force_resolution[1])
            sx = wt / width
            sy = ht / height
            fx *= sx
            fy *= sy
            cx *= sx
            cy *= sy
            width, height = wt, ht

        FovX = 2.0 * np.arctan(width / (2.0 * fx))
        FovY = 2.0 * np.arctan(height / (2.0 * fy))
        cam_intr = np.array([fx, fy, cx, cy], dtype=np.float32)

        print("[Intrinsics] Using COLMAP intrinsics from:", cameras_intrinsic_file)
        print(f"[K] fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}, res={width}x{height}")

        model_tag = f"COLMAP_{model}"

    # ---------------- 2) pick a valid image path for Camera init ----------------
    valid_img = None
    for fn in sorted(os.listdir(images_dir)):
        if fn.lower().endswith((".png", ".jpg", ".jpeg")) and ("sky_mask" not in fn):
            valid_img = os.path.join(images_dir, fn)
            break
    if valid_img is None:
        raise FileNotFoundError(f"[BlenderExtrinsics] No image found in images_dir={images_dir}. "
                                f"Camera init requires an existing image_path.")

    # ---------------- 3) load mesh/world alignment M (ONCE) ----------------
    if mesh_transform_json is not None and os.path.exists(mesh_transform_json):
        with open(mesh_transform_json, "r", encoding="utf-8") as f:
            mt = json.load(f)
        if "mesh_to_world" not in mt:
            raise ValueError(f"[BlenderExtrinsics] {mesh_transform_json} has no 'mesh_to_world'")
        M = np.array(mt["mesh_to_world"], dtype=np.float64)
        print("[BlenderExtrinsics] Using mesh_to_world alignment from:", mesh_transform_json)
    else:
        M = np.eye(4, dtype=np.float64)

    # Blender -> COLMAP/OpenCV camera coords flip
    R_flip = np.eye(4, dtype=np.float64)
    R_flip[1, 1] = -1.0
    R_flip[2, 2] = -1.0

    # ---------------- 4) build CameraInfo list ----------------
    cam_infos = []
    for idx, c in enumerate(cameras):
        w2c_bl = np.array(c["world_to_camera"], dtype=np.float64)

        # Align blender world -> QGS/COLMAP world
        w2c_qgs = w2c_bl @ M

        # Convert camera axis convention Blender -> OpenCV/COLMAP
        w2c = R_flip @ w2c_qgs

        # R stored transposed in this codebase
        R = np.transpose(w2c[:3, :3]).astype(np.float32)
        T = w2c[:3, 3].astype(np.float32)

        image_name = f"blender_{idx:04d}"
        image_path = valid_img  # just for initialization

        cam_info = CameraInfo(
            uid=idx,
            R=R,
            T=T,
            FovY=float(FovY),
            FovX=float(FovX),
            image_path=image_path,
            image_name=image_name,
            image_type=dataset.type,
            width=width,
            height=height,
            downsample=1.0,             # 关键：避免 Camera 内部再缩放
            cam_intr=cam_intr,
            use_alpha=dataset.use_alpha
        )
        cam_infos.append(cam_info)

    print(f"\n[OK] Blender cameras loaded: {len(cam_infos)} views; intrinsics={model_tag}; "
          f"resolution={width}x{height}; using image={os.path.basename(valid_img)}")

    return cam_infos


import os
import json
import numpy as np
from pathlib import Path
from utils.graphics_utils import focal2fov
from scene.colmap_loader import rotmat2qvec, qvec2rotmat  # 如果你没用到可删
# 你文件里已经有 CameraInfo NamedTuple，无需重复定义

def _pick_any_image(images_dir):
    exts = (".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG")
    if images_dir is None:
        return None
    if not os.path.isdir(images_dir):
        return None
    for fn in sorted(os.listdir(images_dir)):
        if ("sky_mask" in fn) or (not fn.endswith(exts)):
            continue
        return os.path.join(images_dir, fn)
    return None

def readBlenderCamerasWithBlenderIntrinsics(
    images_dir,
    dataset,
    blender_extrinsics_json,
    blender_intrinsics_json,
    mesh_transform_json=None,
    force_resolution=None,
    max_views=-1
):
    """
    Build CameraInfo list from:
      - Blender extrinsics JSON: {"cameras":[{"world_to_camera":[[...]] , "name":...}, ...]}
      - Blender intrinsics JSON: {"width":..., "height":..., "fxfy_cxcy":[fx,fy,cx,cy], ...}

    Args:
      images_dir: folder for picking a dummy image_path (for Camera init). Can be your colmap images folder.
      dataset: config.dataset object (must have use_alpha, type)
      blender_extrinsics_json: path to single_view_extrinsics.json (or multi-view)
      blender_intrinsics_json: path to blender_intrinsics_ORIGINAL.json
      mesh_transform_json: optional path to mesh_transform.json, used to align blender-world to QGS/COLMAP world
      force_resolution: optional (W,H) to override intrinsics resolution (will scale K accordingly)
      max_views: render only first N cameras if >0

    Returns:
      List[CameraInfo]
    """

    # ---------- load intrinsics ----------
    with open(blender_intrinsics_json, "r", encoding="utf-8") as f:
        intr = json.load(f)

    w = int(intr["width"])
    h = int(intr["height"])
    fx, fy, cx, cy = map(float, intr["fxfy_cxcy"])

    # Optional override output resolution: scale K accordingly
    if force_resolution is not None:
        wt, ht = int(force_resolution[0]), int(force_resolution[1])
        sx = wt / w
        sy = ht / h
        fx *= sx
        fy *= sy
        cx *= sx
        cy *= sy
        w, h = wt, ht

    # Compute FoV from Blender K
    FovX = 2.0 * np.arctan(w / (2.0 * fx))
    FovY = 2.0 * np.arctan(h / (2.0 * fy))

    cam_intr = np.array([fx, fy, cx, cy], dtype=np.float32)

    # ---------- load extrinsics ----------
    with open(blender_extrinsics_json, "r", encoding="utf-8") as f:
        bex = json.load(f)

    cameras = bex.get("cameras", [])
    if len(cameras) == 0:
        raise ValueError(f"[readBlenderCamerasWithBlenderIntrinsics] No 'cameras' in {blender_extrinsics_json}")

    if max_views is not None and max_views > 0:
        cameras = cameras[:max_views]

    # ---------- load optional mesh/world alignment M ----------
    if mesh_transform_json is not None and os.path.exists(mesh_transform_json):
        with open(mesh_transform_json, "r", encoding="utf-8") as f:
            mt = json.load(f)
        # 你之前导出的字段是 mesh_to_world
        if "mesh_to_world" in mt:
            M = np.array(mt["mesh_to_world"], dtype=np.float64)
        elif "mesh_to_world" in bex:
            M = np.array(bex["mesh_to_world"], dtype=np.float64)
        else:
            raise ValueError("mesh_transform_json exists but does not contain 'mesh_to_world'")
        print("[BlenderIntrinsics] Using mesh_to_world alignment from:", mesh_transform_json)
    else:
        M = np.eye(4, dtype=np.float64)

    # ---------- Blender -> COLMAP/OpenCV axes flip (Blender camera: +Y up, -Z forward) ----------
    R_flip = np.eye(4, dtype=np.float64)
    R_flip[1, 1] = -1.0
    R_flip[2, 2] = -1.0

    # ---------- pick a dummy image path ----------
    dummy_image = _pick_any_image(images_dir)
    if dummy_image is None:
        # 如果没有images_dir也没关系，但你Camera类会读图，所以最好给一个存在的图
        raise FileNotFoundError(f"[readBlenderCamerasWithBlenderIntrinsics] Cannot find any image in {images_dir}")

    cam_infos = []
    for idx, c in enumerate(cameras):
        w2c_bl = np.array(c["world_to_camera"], dtype=np.float64)

        # Align blender world -> QGS/COLMAP world using M (same as你之前成功的版本)
        w2c_qgs = w2c_bl @ M

        # Convert camera axis convention Blender->OpenCV/COLMAP
        w2c = R_flip @ w2c_qgs

        R = np.transpose(w2c[:3, :3]).astype(np.float32)
        T = w2c[:3, 3].astype(np.float32)

        cam_name = c.get("name", f"blender_{idx:04d}")
        image_name = f"blender_{idx:04d}"  # 输出文件名用这个，不依赖原名

        cam_infos.append(CameraInfo(
            uid=idx,
            R=R,
            T=T,
            FovY=float(FovY),
            FovX=float(FovX),
            image_path=dummy_image,
            image_name=image_name,
            image_type=dataset.type,
            width=w,
            height=h,
            downsample=1.0,               # 关键：以 Blender 内参为准，避免 Camera 内部缩放
            cam_intr=cam_intr,
            use_alpha=dataset.use_alpha
        ))

    print(f"[OK] Blender cameras loaded: {len(cam_infos)} views; resolution={w}x{h}; using image={os.path.basename(dummy_image)}")
    return cam_infos


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readMultiScale(path, white_background,split, only_highres=False):
    cam_infos = []
    
    print("read split:", split)
    with open(os.path.join(path, 'metadata.json'), 'r') as fp:
        meta = json.load(fp)[split]
        
    meta = {k: np.array(meta[k]) for k in meta}
    
    # should now have ['pix2cam', 'cam2world', 'width', 'height'] in self.meta
    for idx, relative_path in enumerate(meta['file_path']):
        if only_highres and not relative_path.endswith("d0.png"):
            continue
        image_path = os.path.join(path, relative_path)
        image_name = Path(image_path).stem
        
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = meta["cam2world"][idx]
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image = Image.open(image_path)

        im_data = np.array(image.convert("RGBA"))

        bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

        fovx = focal2fov(meta["focal"][idx], image.size[0])
        fovy = focal2fov(meta["focal"][idx], image.size[1])
        FovY = fovy 
        FovX = fovx

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                        image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
    return cam_infos


def readMultiScaleNerfSyntheticInfo(path, white_background, eval, load_allres=False):
    print("Reading train from metadata.json")
    train_cam_infos = readMultiScale(path, white_background, "train", only_highres=(not load_allres))
    print("number of training images:", len(train_cam_infos))
    print("Reading test from metadata.json")
    test_cam_infos = readMultiScale(path, white_background, "test", only_highres=False)
    print("number of testing images:", len(test_cam_infos))
    if not eval:
        print("adding test cameras to training")
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Multi-scale": readMultiScaleNerfSyntheticInfo,
}