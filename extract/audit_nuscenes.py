import json
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent directory to path to find confidence_scorer
sys.path.append(str(Path(__file__).resolve().parent.parent))
from confidence_scorer.scorer import ConfidenceScorer

def main():
    # 1. Data Loading
    with open('unified_nuscenes_mini.json', 'r') as f:
        data = json.load(f)
        
    scorer = ConfidenceScorer()
    
    total_scenes = len(data)
    total_frames = 0
    total_objects = 0
    dead_frames = []
    
    max_ego_vel = -float('inf')
    min_ego_vel = float('inf')
    
    # Store data for plotting: scene_id -> {"conf": [], "chamfer": []}
    plot_data = {}
    
    # 2. Batch Scoring & 3. Audit Logging
    for scene in data:
        scene_id = scene['scene_id']
        plot_data[scene_id] = {"conf": [], "chamfer": []}
        
        frames = scene['frame_list']
        total_frames += len(frames)
        
        for t in range(len(frames)):
            frame = frames[t]
            objs = frame['object_list']
            total_objects += len(objs)
            
            # Check for dead frames
            if len(objs) == 0:
                dead_frames.append((scene_id, frame['frame_id']))
                
            # Track drift (ego_vel)
            vel = frame['ego_vel']
            if vel > max_ego_vel: max_ego_vel = vel
            if vel < min_ego_vel: min_ego_vel = vel
            
            # Record chamfer distance
            plot_data[scene_id]["chamfer"].append(frame["chamfer_distance"])
            
            # Calculate Confidence Score vs t-1
            if t == 0:
                # First frame has no t-1, we mimic empty t-1
                conf = scorer.calculate_score(objs, [])['confidence_score']
            else:
                prev_objs = frames[t-1]['object_list']
                conf = scorer.calculate_score(objs, prev_objs)['confidence_score']
                
            plot_data[scene_id]["conf"].append(conf)

    # Print summary report
    print("\n" + "="*50)
    print("DATA AUDIT REPORT")
    print("="*50)
    print(f"Total Scenes Processed : {total_scenes}")
    print(f"Total Frames Processed : {total_frames}")
    avg_objs = total_objects / total_frames if total_frames > 0 else 0
    print(f"Average Objects/Frame  : {avg_objs:.2f}")
    
    print("\n--- Drift Check ---")
    print(f"Min Ego Velocity       : {min_ego_vel:.4f} m/s")
    print(f"Max Ego Velocity       : {max_ego_vel:.4f} m/s")
    
    print("\n--- Dead Frames Check ---")
    if not dead_frames:
        print("No dead frames found. All frames have >= 1 object.")
    else:
        print(f"Found {len(dead_frames)} dead frames:")
        for sid, fid in dead_frames[:10]:
            print(f"  Scene: {sid[:8]}..., Frame: {fid}")
        if len(dead_frames) > 10:
            print(f"  ... and {len(dead_frames)-10} more.")
            
    print("="*50 + "\n")
    
    # 4. Time-Series Visualization
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(15, 20))
    axes = axes.flatten()
    
    for idx, (scene_id, metrics) in enumerate(plot_data.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        conf_arr = np.array(metrics["conf"])
        chamfer_arr = np.array(metrics["chamfer"])
        
        # Normalize chamfer distance to 0-1 range per scene for visual comparison
        if chamfer_arr.max() > chamfer_arr.min():
            chamfer_norm = (chamfer_arr - chamfer_arr.min()) / (chamfer_arr.max() - chamfer_arr.min())
        else:
            chamfer_norm = np.zeros_like(chamfer_arr)
            
        frames_x = np.arange(len(conf_arr))
        
        ax.plot(frames_x, conf_arr, 'b-', linewidth=2, label="Confidence Score (Harmonic)")
        ax.plot(frames_x, chamfer_norm, 'r--', linewidth=1.5, alpha=0.7, label="Chamfer Dist (Norm)")
        
        ax.set_title(f"Scene: {scene_id}", fontsize=10)
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Score / Normalized Distance")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=':', alpha=0.6)
        if idx == 0:
            ax.legend()
            
    plt.tight_layout()
    output_img = "nuscenes_similarity_trends.png"
    plt.savefig(output_img, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization to '{output_img}'")

if __name__ == "__main__":
    main()
