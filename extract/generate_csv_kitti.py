import json
import csv
import math
import sys
import os
from pathlib import Path

# Add parent directory to path to find confidence_scorer
sys.path.append(str(Path(__file__).resolve().parent.parent))
from confidence_scorer.scorer import ConfidenceScorer

def calculate_avg_dist(objects: list) -> float:
    if not objects:
        return 0.0
    total_dist = 0.0
    for obj in objects:
        x, y = obj['bbox'][0], obj['bbox'][1]
        dist = math.sqrt(x**2 + y**2)
        total_dist += dist
    return total_dist / len(objects)

def calculate_fastest_vel(objects: list) -> float:
    if not objects:
        return 0.0
    max_v = 0.0
    for obj in objects:
        v_vec = obj.get('velocity', [0.0, 0.0, 0.0])
        v_scalar = math.sqrt(sum(x**2 for x in v_vec))
        if v_scalar > max_v:
            max_v = v_scalar
    return max_v

def calculate_extreme_distances(objects: list) -> tuple:
    if not objects:
        return 0.0, 0.0
    dists = []
    for obj in objects:
        x, y = obj['bbox'][0], obj['bbox'][1]
        dist = math.sqrt(x**2 + y**2)
        dists.append(dist)
    return min(dists), max(dists)

def safe_float(val: float) -> float:
    if not math.isfinite(val):
        return 0.0
    return val

def main():
    scorer = ConfidenceScorer()
    input_file = 'data/json/unified_kitti.json'

    output_dir = Path('data/csv')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'kitti_training_data.csv'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r') as f:
        data = json.load(f)
        
    dataset = []
    for scene in data:
        frames = scene['frame_list']
        for t in range(1, len(frames)):
            frame_t = frames[t]
            frame_prev = frames[t-1]
            
            chamfer_dist = frame_t['chamfer_distance']
            ego_vel = frame_t['ego_vel']
            objs_t = frame_t['object_list']
            objs_prev = frame_prev['object_list']
            
            obj_count = len(objs_t)
            avg_dist = calculate_avg_dist(objs_t)
            fastest_obj_vel = calculate_fastest_vel(objs_prev)
            nearest_obj_dist, farthest_obj_dist = calculate_extreme_distances(objs_prev)
            
            target_confidence = scorer.calculate_score(objs_t, objs_prev)['confidence_score']
            
            row = [
                safe_float(chamfer_dist),
                safe_float(ego_vel),
                float(obj_count),
                safe_float(avg_dist),
                safe_float(fastest_obj_vel),
                safe_float(nearest_obj_dist),
                safe_float(farthest_obj_dist),
                safe_float(target_confidence)
            ]
            dataset.append(row)
            
    headers = ["chamfer_dist", "ego_vel", "obj_count", "avg_dist", "fastest_obj_vel", "nearest_obj_dist", "farthest_obj_dist", "target_confidence"]
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(dataset)
        
    print(f"✅ Generated KITTI tabular dataset at '{output_file}'")
    print(f"Total training samples generated: {len(dataset)}")

if __name__ == "__main__":
    main()
