import json
import os
import numpy as np
from src.gnn.preprocessing import FeatureScaler
from phase9_geant.parse_geant import parse_all_matrices, parse_topology

def main():
    out_dir = "data/processed/geant"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Blocks
    blocks = {
        "train": (0, 8020),
        "val": (8020, 9740),
        "test": (9740, 11460)
    }
    
    valid_mask = np.load(os.path.join(out_dir, "valid_mask.npy"))
    
    window_starts = {"train": [], "val": [], "test": []}
    counts = {}
    
    # Check window starts
    for name, (start, end) in blocks.items():
        starts = []
        for i in range(start, end - 6 + 1):
            if np.all(valid_mask[i:i+6]):
                starts.append(i)
        window_starts[name] = starts
        counts[name] = len(starts)
        
    print("Window counts:", counts)
    if counts["train"] != 7344 or counts["val"] != 1715 or counts["test"] != 1651:
        print("ERROR: Window counts differ from expected!")
        return
        
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    # Fast date retrieval
    import glob
    DATA_DIR = 'data/raw/geant_topology/directed-geant-uhlig-15min-over-4months-ALL-native'
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "demandMatrix-geant-uhlig-15min-*.txt")))
    
    def get_date(idx):
        import re
        m = re.search(r"15min-(\d{8}-\d{4})\.txt", all_files[idx])
        return m.group(1) if m else str(idx)
    
    split_info = {
        "blocks": {
            name: {
                "start_index": start,
                "end_index": end,
                "start_date": get_date(start),
                "end_date": get_date(end-1),
                "window_count": len(window_starts[name]),
                "window_starts": window_starts[name]
            }
            for name, (start, end) in blocks.items()
        }
    }
    
    split_path = os.path.join(out_dir, "split.json")
    with open(split_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"Saved split JSON to {split_path}")
        
    # 3. Collinearity
    traffic = np.load(os.path.join(out_dir, "link_series_traffic.npy"))
    utilization = np.load(os.path.join(out_dir, "link_series_utilization.npy"))
    
    v_traffic = traffic[valid_mask].flatten()
    v_util = utilization[valid_mask].flatten()
    
    corr = np.corrcoef(v_traffic, v_util)[0, 1]
    
    # max abs dev
    expected_traffic = v_util * 40000.0
    max_dev = np.max(np.abs(v_traffic - expected_traffic))
    
    print(f"Collinearity traffic vs utilization: corr={corr:.6f}, max_abs_dev={max_dev:.6e}")
    
    # 4. Scaler on train windows' steps
    # We collect all distinct steps that appear in any valid train window.
    # Actually, the prompt says "Fit the scaler on TRAIN windows' steps only (valid, unflagged)."
    train_starts = window_starts["train"]
    train_step_indices = set()
    for s in train_starts:
        for offset in range(6):
            train_step_indices.add(s + offset)
            
    train_step_indices = sorted(list(train_step_indices))
    train_utils = utilization[train_step_indices] # shape (N, E)
    # We fit the scaler on the flat utilization values or per-edge? "Input feature = utilization only"
    # Wait, scaler fits on the feature vector. Since it's just utilization, the feature vector is size 1.
    # FeatureScaler does values.mean(axis=0) and values.std(axis=0). 
    # If we reshape it to (N*E, 1) to fit a single global mean/std for utilization?
    # No, typically Graph models scale per feature globally. The input to scaler should be (N_items, num_features).
    # Since num_features = 1, we pass shape (N_items, 1).
    flat_train_utils = train_utils.flatten().reshape(-1, 1)
    
    scaler = FeatureScaler()
    scaler.fit(flat_train_utils)
    
    print(f"Scaler stats (Train steps: {len(train_step_indices)}): mean={float(scaler.mean_[0]):.6e}, std={float(scaler.scale_[0]):.6e}")
    
    # save scaler parameters
    scaler_dict = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist()
    }
    with open(os.path.join(out_dir, "scaler.json"), "w") as f:
        json.dump(scaler_dict, f, indent=2)
        
    # 6. Sensitivity on CLEAN data (valid unflagged steps)
    # The valid_mask excludes flagged steps already
    util_hop = np.load(os.path.join(out_dir, "link_series_utilization_hop.npy"))
    
    v_util_hop = util_hop[valid_mask]
    v_util_dist = utilization[valid_mask]
    
    diff = np.abs(v_util_dist - v_util_hop)
    mean_abs_diff = np.mean(diff)
    corr_clean = np.corrcoef(v_util_dist.flatten(), v_util_hop.flatten())[0, 1]
    
    print(f"Sensitivity on CLEAN data: mean_abs_diff={mean_abs_diff:.6f}, corr={corr_clean:.6f}")
    
if __name__ == '__main__':
    main()
