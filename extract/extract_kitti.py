import json
import os
import glob
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial import cKDTree

# ── Configuration ─────────────────────────────────────────────────────────────
DATAROOT = "data/kitti/2011_09_26"
OUTPUT = "data/json/unified_kitti.json"
DOWNSAMPLE = 500
RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_velodyne_points(path: str) -> np.ndarray:
    """Load a KITTI .bin velodyne file → (N, 3) xyz array."""
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return pts[:, :3]

def downsample(pts: np.ndarray, n: int) -> np.ndarray:
    """Randomly downsample to at most n points."""
    if len(pts) <= n:
        return pts
    idx = rng.choice(len(pts), size=n, replace=False)
    return pts[idx]

def chamfer_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """Symmetric Chamfer distance between two (N,3) point clouds."""
    if pts_a is None or pts_b is None:
        return 0.0
    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)
    d_ab, _ = tree_b.query(pts_a, k=1)
    d_ba, _ = tree_a.query(pts_b, k=1)
    return float(np.mean(d_ab) + np.mean(d_ba))

def parse_oxts_data(file_path: str):
    """
    Parses KITTI OXTS .txt file and returns ego_vel and ego_accel_magnitude.
    - Velocity Indices: 8, 9, 10 (vf, vl, vu)
    - Acceleration Indices: 14, 15, 16 (af, al, au)
    """
    with open(file_path, 'r') as f:
        line = f.readline()
        if not line:
            return 0.0, 0.0
        data = [float(x) for x in line.split()]

        # Velocity
        vf, vl, vu = data[8], data[9], data[10]
        ego_vel = np.sqrt(vf**2 + vl**2 + vu**2)

        # Acceleration
        af, al, au = data[14], data[15], data[16]
        ego_accel_magnitude = np.sqrt(af**2 + al**2 + au**2)
        
    return float(ego_vel), float(ego_accel_magnitude)

def parse_tracklets(xml_path: str) -> list:
    """
    Parses KITTI tracklet_labels.xml and returns a list of tracklets.
    """
    if not xml_path or not os.path.exists(xml_path):
        print(f"    ⚠️ Warning: Tracklet file not found at {xml_path}")
        return []
        
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        tracklets = []
        # Find elements under <tracklets>
        # The XML structure has <tracklets> as a child of <boost_serialization>
        tracklets_elem = root.find('tracklets')
        if tracklets_elem is None:
            return []

        for item in tracklets_elem.findall('item'):
            obj_type = item.find('objectType').text
            h = float(item.find('h').text)
            w = float(item.find('w').text)
            l = float(item.find('l').text)
            first_frame = int(item.find('first_frame').text)
            
            poses = []
            poses_elem = item.find('poses')
            if poses_elem is not None:
                for pose_item in poses_elem.findall('item'):
                    tx = float(pose_item.find('tx').text)
                    ty = float(pose_item.find('ty').text)
                    tz = float(pose_item.find('tz').text)
                    rz = float(pose_item.find('rz').text) # Rotation around Z (yaw)
                    poses.append([tx, ty, tz, rz])
                
            tracklets.append({
                'label': obj_type,
                'h': h, 'w': w, 'l': l,
                'first_frame': first_frame,
                'poses': poses
            })
        return tracklets
    except Exception as e:
        print(f"    Error parsing tracklets: {e}")
        return []

def get_objects_from_tracklets(tracklets: list, frame_idx: int) -> list:
    """Extract objects for a specific frame from parsed tracklets."""
    objects = []
    for i, trk in enumerate(tracklets):
        # Check if frame_idx is within the tracklet range
        if trk['first_frame'] <= frame_idx < trk['first_frame'] + len(trk['poses']):
            pose_idx = frame_idx - trk['first_frame']
            p = trk['poses'][pose_idx] # [tx, ty, tz, rz]
            
            # Calculate velocity if possible
            velocity = [0.0, 0.0, 0.0]
            if pose_idx > 0:
                p_prev = trk['poses'][pose_idx - 1]
                # KITTI is ~10Hz
                velocity = [
                    (p[0] - p_prev[0]) / 0.1,
                    (p[1] - p_prev[1]) / 0.1,
                    (p[2] - p_prev[2]) / 0.1
                ]
            
            objects.append({
                "obj_id": f"tracklet_{i}",
                "label": trk['label'],
                "bbox": [p[0], p[1], p[2], trk['l'], trk['w'], trk['h'], p[3]],
                "velocity": velocity
            })
    return objects

def extract_drive(drive_path: str, tracklet_path: str = None) -> dict:
    """Extract a single KITTI drive sequence."""
    drive_id = os.path.basename(drive_path)
    print(f"  → Processing drive: {drive_id}")
    
    # Handle nested data structure
    data_path = drive_path
    # Look for oxts and velodyne_points even if nested
    oxts_search = glob.glob(os.path.join(drive_path, "**/oxts/data/*.txt"), recursive=True)
    lidar_search = glob.glob(os.path.join(drive_path, "**/velodyne_points/data/*.bin"), recursive=True)
    
    if not oxts_search or not lidar_search:
        print(f"    Error: Could not find OXTS or Lidar data in {drive_path}")
        return None
        
    oxts_files = sorted(oxts_search)
    lidar_files = sorted(lidar_search)
    
    # Parse tracklets if provided
    tracklets = parse_tracklets(tracklet_path) if tracklet_path else []
    if tracklets:
        print(f"    Loaded {len(tracklets)} tracklets.")
    else:
        print("    No tracklets found, object list will be empty.")

    scene_out = {
        "scene_id": drive_id,
        "dataset_id": "KITTI",
        "frame_list": []
    }
    
    # Ensure they are synced (KITTI sync usually has same counts)
    num_frames = min(len(oxts_files), len(lidar_files))
    
    prev_pts = None
    
    for i in range(num_frames):
        oxts_path = oxts_files[i]
        lidar_path = lidar_files[i]
        frame_id = os.path.splitext(os.path.basename(oxts_path))[0]
        
        # Ego info
        vel, accel = parse_oxts_data(oxts_path)
        
        # Lidar / Chamfer distance
        curr_pts = downsample(load_velodyne_points(lidar_path), DOWNSAMPLE)
        cd = chamfer_distance(curr_pts, prev_pts) if prev_pts is not None else 0.0
        
        # Extract objects from tracklets
        object_list = get_objects_from_tracklets(tracklets, i)
        
        # Assemble frame
        frame = {
            "frame_id": frame_id,
            "chamfer_distance": cd,
            "ego_vel": vel,
            "ego_accel_raw": accel,
            "object_list": object_list
        }
        scene_out["frame_list"].append(frame)
        prev_pts = curr_pts
        
    return scene_out

def main():
    if not os.path.exists(DATAROOT):
        print(f"Error: Dataroot {DATAROOT} not found.")
        return

    drive_sequences = sorted([d for d in glob.glob(os.path.join(DATAROOT, "*_sync")) if os.path.isdir(d)])
    print(f"Found {len(drive_sequences)} possible drive sequences.")
    
    all_scenes = []
    for drive_path in drive_sequences:
        # Find corresponding tracklets
        drive_name = os.path.basename(drive_path)
        # e.g. 2011_09_26_drive_0002_sync -> 2011_09_26_drive_0002_tracklets
        tracklet_dir_name = drive_name.replace("_sync", "_tracklets")
        tracklet_root = os.path.join(DATAROOT, tracklet_dir_name)
        
        tracklet_path = None
        if os.path.exists(tracklet_root):
            # Look for tracklet_labels.xml inside (handling nesting)
            xml_search = glob.glob(os.path.join(tracklet_root, "**/tracklet_labels.xml"), recursive=True)
            if xml_search:
                tracklet_path = xml_search[0]
        
        scene_data = extract_drive(drive_path, tracklet_path)
        if scene_data:
            all_scenes.append(scene_data)
            print(f"    - Extracted {len(scene_data['frame_list'])} frames.")
        
    with open(OUTPUT, "w") as f:
        json.dump(all_scenes, f, indent=2)
        
    print(f"\n✅ Saved {len(all_scenes)} scenes to '{OUTPUT}'")

if __name__ == "__main__":
    main()
