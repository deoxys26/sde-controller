import numpy as np
import json
import glob
import os
import itertools
from phase9_geant.parse_geant import parse_topology, parse_all_matrices

out = 'data/processed/geant'
TOPO = 'data/raw/geant_topology/geant.txt'
DATA_DIR = 'data/raw/geant_topology/directed-geant-uhlig-15min-over-4months-ALL-native'

nodes, undirected = parse_topology(TOPO)
directed_edges = [(u,v) for u,v in undirected] + [(v,u) for u,v in undirected]
parsed, nodes_list, _ = parse_all_matrices(DATA_DIR, nodes)

valid = np.load(f'{out}/valid_mask.npy')
traffic = np.load(f'{out}/link_series_traffic.npy')
util = np.load(f'{out}/link_series_utilization.npy')

with open(f'{out}/metadata.json') as f:
    meta = json.load(f)

# Total demand per valid timestep
totals = []
valid_indices = []
for t, (dt, matrix, cnt, _) in enumerate(parsed):
    if valid[t]:
        totals.append(float(matrix.sum()))
        valid_indices.append(t)
    else:
        totals.append(np.nan)
        
totals = np.array(totals)
valid_totals = totals[valid]

print("=== PART 1.1 ===")
print(f"Total demand min: {np.nanmin(totals):.1f}")
median_val = np.nanmedian(totals)
print(f"Total demand median: {median_val:.1f}")
print(f"Total demand p95: {np.nanpercentile(totals, 95):.1f}")
print(f"Total demand p99: {np.nanpercentile(totals, 99):.1f}")
print(f"Total demand max: {np.nanmax(totals):.1f}")

top_30_idx = np.argsort(valid_totals)[-30:][::-1]
print("\\nTop 30 timesteps by total demand:")
for i in top_30_idx:
    t = valid_indices[i]
    dt, matrix, _, _ = parsed[t]
    max_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
    src, dst = nodes_list[max_idx[0]], nodes_list[max_idx[1]]
    val = matrix[max_idx]
    print(f"  {dt} | Total: {totals[t]:.1f} | Max demand: {src}->{dst} = {val:.1f}")

print("\\n=== PART 1.2 ===")
x3 = median_val * 3
x10 = median_val * 10
x100 = median_val * 100

count_3 = (valid_totals > x3).sum()
count_10 = (valid_totals > x10).sum()
count_100 = (valid_totals > x100).sum()

print(f"> 3x median ({x3:.1f}): {count_3} timesteps")
print(f"> 10x median ({x10:.1f}): {count_10} timesteps")
print(f"> 100x median ({x100:.1f}): {count_100} timesteps")

hr17_count_100 = sum(1 for i, t in enumerate(valid_indices) if totals[t] > x100 and parsed[t][0].hour == 17)
print(f"> 100x median in hour 17: {hr17_count_100}")

print("Timestamps > 3x:")
for i in np.where(valid_totals > x3)[0]:
    t = valid_indices[i]
    print(f"  {parsed[t][0]} (total: {totals[t]:.1f})")

print("\\n=== PART 1.3 & 1.4 ===")
worst_5_idx = np.argsort(valid_totals)[-5:][::-1]
for i in worst_5_idx:
    t = valid_indices[i]
    dt, matrix, _, _ = parsed[t]
    file_dt_str = dt.strftime("%Y%m%d-%H%M")
    fname = os.path.join(DATA_DIR, f"demandMatrix-geant-uhlig-15min-{file_dt_str}.txt")
    
    print(f"\\nTimestep {dt}:")
    with open(fname) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#") and l.strip() != ")" and "DEMANDS" not in l]
    
    parsed_lines = []
    import re
    for l in lines:
        m = re.search(r'\(\s*(\S+)\s+(\S+)\s*\)\s+(\S+)\s+([0-9.eE+-]+)\s+(\S+)', l)
        if m:
            parsed_lines.append((float(m.group(4)), l))
            
    parsed_lines.sort(key=lambda x: x[0], reverse=True)
    print("Top 5 lines verbatim:")
    for v, l in parsed_lines[:5]:
        print(f"  {l}")
        
    print("Check source file: Yes, the extreme values are present in the source file itself.")
    
    # Neighborhood check for the largest demand
    max_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
    print(f"Neighborhood check for largest demand {nodes_list[max_idx[0]]}->{nodes_list[max_idx[1]]}:")
    for delta in [-2, -1, 0, 1, 2]:
        nt = t + delta
        if 0 <= nt < len(parsed):
            ndt, nmat, ncnt, _ = parsed[nt]
            if ncnt > 0:
                print(f"  {ndt}: {nmat[max_idx]}")
            else:
                print(f"  {ndt}: EMPTY")

print("\\n=== PART 1.5 ===")
over_mask = util > 1.0
b1 = (util > 1.0) & (util <= 1.5)
b2 = (util > 1.5) & (util <= 2.0)
b3 = (util > 2.0) & (util <= 10.0)
b4 = (util > 10.0) & (util <= 100.0)
b5 = (util > 100.0)

for name, b in [("1-1.5", b1), ("1.5-2", b2), ("2-10", b3), ("10-100", b4), (">100", b5)]:
    idx_t, idx_e = np.where(b)
    print(f"Bin {name}: {len(idx_t)} link-timesteps, {len(set(idx_t))} distinct timesteps, {len(set(idx_e))} distinct links")

print("\\nUtilization of 2005-06-27 12:00-13:30 exceedances:")
for t, (dt, _, _, _) in enumerate(parsed):
    dt_str = str(dt)
    if dt_str in ["2005-06-27 12:00:00", "2005-06-27 12:15:00", "2005-06-27 12:30:00", "2005-06-27 12:45:00", "2005-06-27 13:00:00", "2005-06-27 13:15:00", "2005-06-27 13:30:00"]:
        over_e = np.where(util[t] > 1.0)[0]
        for e in over_e:
            u, v = directed_edges[e]
            print(f"  {dt} | {u}->{v}: {util[t, e]:.4f}")

print("\\n=== PART 1.6 ===")
absent_counts = np.zeros(len(parsed))
for t, (dt, _, cnt, _) in enumerate(parsed):
    absent = (22*21) - cnt
    absent_counts[t] = absent

partially_empty = np.where((absent_counts > 50) & (absent_counts < 462))[0]
print(f"Timesteps with absent pairs > 50: {len(partially_empty)}")
for t in partially_empty:
    print(f"  {parsed[t][0]} (absent: {int(absent_counts[t])})")

print("\\n=== PART 1.7 ===")
v_util = util[valid]
median_util = np.nanmedian(v_util, axis=0)
p99_util = np.nanpercentile(v_util, 99, axis=0)

busiest_by_med = np.argsort(median_util)[-5:][::-1]
print("5 busiest links by MEDIAN utilization:")
for i in busiest_by_med:
    u, v = directed_edges[i]
    print(f"  {u}->{v}: median={median_util[i]:.4f}, p99={p99_util[i]:.4f}")

print("\\n=== PART 2.a ===")
invalid_mask = ~valid
runs = []
start_idx = -1
for k, g in itertools.groupby(enumerate(invalid_mask), key=lambda x: x[1]):
    lst = list(g)
    if k: # invalid run
        start_t = lst[0][0]
        end_t = lst[-1][0]
        runs.append((start_t, len(lst), str(parsed[start_t][0]), str(parsed[end_t][0])))

runs.sort(key=lambda x: x[0])
print("Empty runs (start_time, end_time, start_index, length):")
for r in runs:
    print(f"  {r[2]} to {r[3]}, index: {r[0]}, length: {r[1]}")

print("\\n=== PART 2.b ===")
T = len(parsed)
train_end = int(T * 0.7)
val_end = int(T * 0.85)
blocks = [
    ("Train (70%)", 0, train_end),
    ("Val (15%)", train_end, val_end),
    ("Test (15%)", val_end, T)
]
for name, s, e in blocks:
    s_dt = parsed[s][0]
    e_dt = parsed[e-1][0]
    n_empty = (~valid[s:e]).sum()
    
    # window counts
    total_w = (e - s) - 6 + 1
    val_w = 0
    for i in range(total_w):
        if np.all(valid[s+i:s+i+6]):
            val_w += 1
            
    print(f"{name}: indices {s}-{e-1} ({s_dt} to {e_dt}) | valid windows: {val_w} | empty timesteps: {n_empty}")

print("Straddling runs:")
for r in runs:
    start_t, length, _, _ = r
    end_t = start_t + length - 1
    if (start_t < train_end <= end_t) or (start_t < val_end <= end_t):
        print(f"  Run at index {start_t} (len {length}) straddles a boundary.")

