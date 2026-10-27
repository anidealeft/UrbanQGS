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

import json
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm

from arguments import ModelParams
from scene.colmap_loader import (
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
)
from scene.dataset_readers import getNerfppNorm, sceneLoadTypeCallbacks, storePly
from scene.scene_gaussian_model2_refactored import GaussianModel
from scene.partition_utils import build_regular_grid_blocks, filter_scene_to_block, write_block_manifest
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.system_utils import searchForMaxIteration


def params_from_models(camera_model: np.ndarray):
    K = np.array(
        [[camera_model[0], 0, camera_model[2]],
         [0, camera_model[1], camera_model[3]],
         [0, 0, 1]],
        dtype=np.float32,
    )
    try:
        distortion = camera_model[4:]
    except Exception:
        distortion = np.array([0, 0, 0, 0], dtype=np.float32)
    return K, distortion


class Scene:
    gaussians: GaussianModel

    def __init__(self, config, gaussians: GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        self.config = config
        self.root_dir = config.root_dir
        self.model_path = config.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.img_downsample = self.config.dataset.downsample
        self.config.white_background = False
        self.img_dir = os.path.join(self.root_dir, f"images_undistorted_{self.img_downsample}")
        self.partition_meta = None

        images_prefix = "images" if os.path.exists(os.path.join(self.root_dir, "sparse")) else "train"
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print(f"Loading trained model at iteration {self.loaded_iter}")

        self.train_cameras = {}
        self.test_cameras = {}

        self.load_images(images_prefix)
        self.scene_info = self.load_cam()
        self.scene_info = self._apply_partition_if_needed(self.scene_info)

        if not self.loaded_iter:
            with open(self.scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"), 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if self.scene_info.test_cameras:
                camlist.extend(self.scene_info.test_cameras)
            if self.scene_info.train_cameras:
                camlist.extend(self.scene_info.train_cameras)
            for idx, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(idx, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w', encoding='utf-8') as file:
                json.dump(json_cams, file, ensure_ascii=False)

        if shuffle:
            random.shuffle(self.scene_info.train_cameras)
            random.shuffle(self.scene_info.test_cameras)

        self.cameras_extent = self.scene_info.nerf_normalization["radius"]
        print(f"cameras_extent {self.cameras_extent}")

        self.multi_view_num = config.gs_model.multi_view_num
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                self.scene_info.train_cameras, resolution_scale, self.config.dataset
            )
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                self.scene_info.test_cameras, resolution_scale, self.config.dataset
            )
            self._compute_nearest_views(resolution_scale)

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                    "point_cloud.ply",
                )
            )
        else:
            self.gaussians.create_from_pcd(self.scene_info.point_cloud, self.cameras_extent)

    def _apply_partition_if_needed(self, scene_info):
        part_cfg = getattr(self.config.dataset, "partition", None)
        if part_cfg is None or not getattr(part_cfg, "enabled", False):
            return scene_info

        grid_x = int(getattr(part_cfg, "grid_x", 1))
        grid_y = int(getattr(part_cfg, "grid_y", 1))
        block_id = int(getattr(part_cfg, "block_id", 0))
        overlap_ratio = float(getattr(part_cfg, "overlap_ratio", 0.10))
        min_cameras = int(getattr(part_cfg, "min_cameras", 8))
        point_pad = float(getattr(part_cfg, "point_pad", 0.15))

        blocks = build_regular_grid_blocks(
            scene_info.train_cameras,
            grid_x=grid_x,
            grid_y=grid_y,
            overlap_ratio=overlap_ratio,
            min_cameras=min_cameras,
        )
        if len(blocks) == 0:
            raise RuntimeError("[Partition] No valid blocks were generated. Reduce min_cameras or adjust grid/overlap.")
        if block_id < 0 or block_id >= len(blocks):
            raise IndexError(f"[Partition] block_id={block_id} out of range, total blocks={len(blocks)}")

        block_train, block_test, block_pcd, block_meta = filter_scene_to_block(
            scene_info.train_cameras,
            scene_info.test_cameras,
            scene_info.point_cloud,
            blocks[block_id],
            point_pad=point_pad,
        )
        if len(block_train) == 0:
            raise RuntimeError(f"[Partition] block_id={block_id} has zero train cameras after filtering.")

        nerf_normalization = getNerfppNorm(block_train)
        ply_path = scene_info.ply_path
        if block_pcd is not None and np.asarray(block_pcd.points).shape[0] > 0:
            ply_path = os.path.join(self.model_path, f"partition_block_{block_id:03d}_input.ply")
            colors = np.asarray(block_pcd.colors)
            if colors.size > 0 and colors.max() <= 1.0:
                colors = np.clip(colors * 255.0, 0, 255)
            storePly(ply_path, np.asarray(block_pcd.points), colors.astype(np.uint8))

        block_meta = dict(block_meta)
        block_meta.update({
            "grid_x": grid_x,
            "grid_y": grid_y,
            "overlap_ratio": overlap_ratio,
            "min_cameras": min_cameras,
            "point_pad": point_pad,
            "train_camera_names": [c.image_name for c in block_train],
            "test_camera_names": [c.image_name for c in block_test],
        })
        write_block_manifest(os.path.join(self.model_path, "block_manifest.json"), block_meta)
        self.partition_meta = block_meta

        print(
            f"[Partition] block_id={block_id} | train_views={len(block_train)} | "
            f"test_views={len(block_test)} | points={0 if block_pcd is None else np.asarray(block_pcd.points).shape[0]}"
        )

        return scene_info._replace(
            point_cloud=block_pcd,
            train_cameras=block_train,
            test_cameras=block_test,
            nerf_normalization=nerf_normalization,
            ply_path=ply_path,
        )

    def _compute_nearest_views(self, resolution_scale):
        print("computing nearest_id")
        cams = self.train_cameras[resolution_scale]
        if len(cams) == 0:
            return

        world_view_transforms = []
        camera_centers = []
        center_rays = []
        for cur_cam in cams:
            cur_cam.nearest_id = []
            cur_cam.nearest_names = []
            world_view_transforms.append(cur_cam.world_view_transform)
            camera_centers.append(cur_cam.camera_center)
            R = torch.tensor(cur_cam.R).float().cuda()
            center_ray = torch.tensor([0.0, 0.0, 1.0]).float().cuda()
            center_ray = center_ray @ R.transpose(-1, -2)
            center_rays.append(center_ray)

        self.world_view_transforms = torch.stack(world_view_transforms)
        camera_centers = torch.stack(camera_centers, dim=0)
        center_rays = torch.stack(center_rays, dim=0)
        center_rays = torch.nn.functional.normalize(center_rays, dim=-1)
        diss = torch.norm(camera_centers[:, None] - camera_centers[None], dim=-1).detach().cpu().numpy()
        tmp = torch.sum(center_rays[:, None] * center_rays[None], dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        angles = (torch.arccos(tmp) * 180.0 / np.pi).detach().cpu().numpy()

        with open(os.path.join(self.model_path, "multi_view.json"), 'w', encoding='utf-8') as file:
            for cam_id, cur_cam in enumerate(cams):
                sorted_indices = np.lexsort((angles[cam_id], diss[cam_id]))
                mask = (
                    (angles[cam_id][sorted_indices] < self.config.dataset.multi_view_max_angle)
                    & (diss[cam_id][sorted_indices] > self.config.dataset.multi_view_min_dis)
                    & (diss[cam_id][sorted_indices] < self.config.dataset.multi_view_max_dis)
                )
                sorted_indices = sorted_indices[mask]
                multi_view_num = min(self.multi_view_num, len(sorted_indices))
                json_d = {"ref_name": cur_cam.image_name, "nearest_name": []}
                for index in sorted_indices[:multi_view_num]:
                    if index == cam_id:
                        continue
                    cur_cam.nearest_id.append(int(index))
                    cur_cam.nearest_names.append(cams[index].image_name)
                    json_d["nearest_name"].append(cams[index].image_name)
                file.write(json.dumps(json_d, separators=(',', ':'), ensure_ascii=False))
                file.write('\n')

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, f"point_cloud/iteration_{iteration}")
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def load_images(self, images):
        if not os.path.exists(self.img_dir):
            self.undist_and_downsample()
        if len(os.listdir(self.img_dir)) != len(os.listdir(os.path.join(self.root_dir, images))):
            self.undist_and_downsample()

    def load_cam(self):
        if os.path.exists(os.path.join(self.root_dir, "sparse")):
            print("Found sparse directory, assuming Colmap data set!")
            scene_info = sceneLoadTypeCallbacks["Colmap"](self.root_dir, self.img_dir, self.config.dataset)
        elif os.path.exists(os.path.join(self.root_dir, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                self.root_dir, self.config.dataset.white_background, self.config.dataset.eval, self.config.dataset
            )
        elif os.path.exists(os.path.join(self.config.source_path, "metadata.json")):
            print("Found metadata.json file, assuming multi scale Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Multi-scale"](
                self.root_dir, self.config.dataset.white_background, self.config.dataset.eval
            )
        else:
            raise AssertionError("Could not recognize scene type!")
        return scene_info

    def undist_and_downsample(self):
        prefix = f"images_undistorted_{self.img_downsample}"
        output_dir = os.path.join(self.root_dir, prefix)
        input_dir = os.path.join(self.root_dir, "images")
        if os.path.exists(os.path.join(self.root_dir, "transforms_train.json")):
            input_dir = os.path.join(self.root_dir, "train")

        os.makedirs(output_dir, exist_ok=True)
        if os.path.exists(os.path.join(self.root_dir, "sparse")):
            models, img_data = self.read_meta()
            for img_idx in tqdm(img_data):
                img_path = os.path.join(input_dir, img_data[img_idx].name)
                output_path = os.path.join(output_dir, img_data[img_idx].name)
                distorted = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                img_wh = int(distorted.shape[1] * self.img_downsample), int(distorted.shape[0] * self.img_downsample)
                if self.config.dataset.undistortion and models[img_data[img_idx].camera_id].model in ["OPENCV"]:
                    try:
                        K, distortion = params_from_models(models[img_data[img_idx].camera_id].params)
                        undistorted = cv2.undistort(distorted, K, distortion)
                    except Exception:
                        undistorted = distorted
                else:
                    undistorted = distorted

                undistorted = cv2.resize(undistorted, (img_wh[0], img_wh[1]))
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, undistorted)
        else:
            input_path = os.path.join(input_dir, os.listdir(input_dir)[0])
            img_shape = Image.open(input_path).size
            target_size = (int(self.img_downsample * img_shape[0]), int(self.img_downsample * img_shape[1]))
            for filename in tqdm(os.listdir(input_dir)):
                if filename.lower().endswith(".png"):
                    input_path = os.path.join(input_dir, filename)
                    output_path = os.path.join(output_dir, filename)
                    with Image.open(input_path) as img:
                        resized_img = img.resize(target_size, Image.LANCZOS)
                        resized_img.save(output_path)

    def read_meta(self):
        try:
            img_data = read_extrinsics_binary(os.path.join(self.root_dir, 'sparse/0/images.bin'))
            cam_data = read_intrinsics_binary(os.path.join(self.root_dir, 'sparse/0/cameras.bin'))
        except Exception:
            img_data = read_extrinsics_text(os.path.join(self.root_dir, 'sparse/0/images.txt'))
            cam_data = read_intrinsics_text(os.path.join(self.root_dir, 'sparse/0/cameras.txt'))
        return cam_data, img_data
