import json
import os
import numpy as np
import pytest

out_dir = "data/processed/geant"

def test_split_geometry():
    with open(os.path.join(out_dir, "split.json")) as f:
        split = json.load(f)["blocks"]
        
    train = split["train"]
    val = split["val"]
    test = split["test"]
    
    assert train["start_index"] == 0
    assert train["end_index"] == 8020
    assert val["start_index"] == 8020
    assert val["end_index"] == 9740
    assert test["start_index"] == 9740
    assert test["end_index"] == 11460
    
    assert train["window_count"] == 7344
    assert val["window_count"] == 1715
    assert test["window_count"] == 1651
    
def test_windows_validity():
    with open(os.path.join(out_dir, "split.json")) as f:
        split = json.load(f)["blocks"]
    
    valid_mask = np.load(os.path.join(out_dir, "valid_mask.npy"))
    
    for name, block in split.items():
        starts = block["window_starts"]
        start_idx = block["start_index"]
        end_idx = block["end_index"]
        
        for s in starts:
            # no window crosses boundary
            assert s >= start_idx
            assert s + 5 < end_idx
            
            # no window contains invalid step
            assert np.all(valid_mask[s:s+6]), f"Window at {s} in {name} contains invalid step"
            
def test_scaler_determinism():
    with open(os.path.join(out_dir, "scaler.json")) as f:
        scaler = json.load(f)
        
    mean = scaler["mean"][0]
    scale = scaler["scale"][0]
    assert np.isclose(mean, 0.03158379, rtol=1e-3), f"Mean {mean} unexpected"
    assert scale > 0
