import json
import os
import numpy as np
from scipy.spatial import cKDTree
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes

# ── Configuration ─────────────────────────────────────────────────────────────
DATAROOT = "data/nuscenes"
VERSION  = "v1.0-mini"
OUTPUT   = "data/json/unified_nuscenes_mini.json"
LIDAR_CHAN = "LIDAR_TOP"
DOWNSAMPLE = 500
RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_pointcloud(path: str) -> np.ndarray:
    """Load a nuScenes .pcd.bin file → (N, 3) xyz array."""
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    return pts[:, :3]


def downsample(pts: np.ndarray, n: int) -> np.ndarray:
    """Randomly downsample to at most n points."""
    if len(pts) <= n:
        return pts
    idx = rng.choice(len(pts), size=n, replace=False)
    return pts[idx]


def chamfer_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """Symmetric Chamfer distance between two (N,3) point clouds."""
    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)
    d_ab, _ = tree_b.query(pts_a, k=1)   # each point in A → nearest in B
    d_ba, _ = tree_a.query(pts_b, k=1)   # each point in B → nearest in A
    return float(np.mean(d_ab) + np.mean(d_ba))


def quat_to_yaw(rotation: list) -> float:
    """Convert quaternion [w, x, y, z] to yaw (rotation about Z axis)."""
    q = Quaternion(rotation)
    # yaw = atan2(2*(w*z + x*y), 1 - 2*(y² + z²))
    yaw = q.yaw_pitch_roll[0]
    return float(yaw)


def get_lidar_sample_data(nusc: NuScenes, sample_token: str):
    """Return the LIDAR_TOP sample_data record for a given sample."""
    sample = nusc.get("sample", sample_token)
    sd_token = sample["data"][LIDAR_CHAN]
    return nusc.get("sample_data", sd_token)


# ── Main extraction ───────────────────────────────────────────────────────────
def extract_scene(nusc: NuScenes, scene: dict) -> dict:
    scene_out = {
        "scene_id":   scene["token"],
        "dataset_id": "nuScenes",
        "frame_list": [],
    }

    first_sample_token = scene["first_sample_token"]
    first_sample = nusc.get("sample", first_sample_token)
    sd_token = first_sample["data"][LIDAR_CHAN]
    
    sd = nusc.get("sample_data", sd_token)
    while sd["prev"]:
        sd = nusc.get("sample_data", sd["prev"])

    prev_pts = None
    prev_ego_pose = None

    while sd:
        # Ego Pose
        ego_pose = nusc.get("ego_pose", sd["ego_pose_token"])
        lidar_path = nusc.get_sample_data_path(sd["token"])
        curr_pts = downsample(load_pointcloud(lidar_path), DOWNSAMPLE)

        # Metrics
        cd = chamfer_distance(curr_pts, prev_pts) if prev_pts is not None else 0.0
        ego_vel = 0.0
        if prev_ego_pose:
            dt = (ego_pose["timestamp"] - prev_ego_pose["timestamp"]) * 1e-6
            if dt > 0:
                delta = np.array(ego_pose["translation"]) - np.array(prev_ego_pose["translation"])
                ego_vel = float(np.linalg.norm(delta) / dt)

        # Box Interpolation
        object_list = []
        boxes = nusc.get_boxes(sd["token"]) 
        
        for box in boxes:
            x, y, z = box.center
            w, l, h = box.wlh
            yaw = quat_to_yaw(box.orientation.elements.tolist())
            
            # Velocity is typically stored in the box if available from the sample
            vel = nusc.box_velocity(box.token) 
            if np.any(np.isnan(vel)):
                vel = [0.0, 0.0, 0.0]
            else:
                vel = vel.tolist()

            object_list.append({
                "obj_id": nusc.get("sample_annotation", box.token)["instance_token"],
                "label":  box.name,
                "bbox":   [x, y, z, w, l, h, yaw],
                "velocity": vel,
                "is_key_frame": sd["is_key_frame"]
            })

        # Assemble
        scene_out["frame_list"].append({
            "frame_id":     sd["token"],
            "timestamp":    sd["timestamp"],
            "is_key_frame": sd["is_key_frame"],
            "chamfer_distance": cd,
            "ego_vel":      ego_vel,
            "object_list":  object_list,
        })

        # Move to NEXT SWEEP
        prev_pts = curr_pts
        prev_ego_pose = ego_pose
        sd = nusc.get("sample_data", sd["next"]) if sd["next"] else None

    return scene_out

def main():
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)

    all_scenes = []
    for i, scene in enumerate(nusc.scene):
        print(f"\n[{i+1}/{len(nusc.scene)}] Processing scene: {scene['name']} ({scene['token']})")
        scene_data = extract_scene(nusc, scene)
        all_scenes.append(scene_data)
        print(f"  → {len(scene_data['frame_list'])} frames extracted")

    with open(OUTPUT, "w") as f:
        json.dump(all_scenes, f, indent=2)

    print(f"\n✅  Saved {len(all_scenes)} scenes to '{OUTPUT}'")


if __name__ == "__main__":
    main()
