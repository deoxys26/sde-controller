import os
import json
import numpy as np
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt

def main():
    out_dir = "data/processed/geant"
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    
    traffic = np.load(os.path.join(out_dir, "link_series_traffic.npy"))
    valid_mask = np.load(os.path.join(out_dir, "valid_mask.npy"))
    
    with open(os.path.join(out_dir, "metadata.json")) as f:
        meta = json.load(f)
        
    start_dt = datetime.strptime(meta["date_range"]["start"], "%Y-%m-%d %H:%M:%S")
    
    total_demand = []
    hourly = defaultdict(list)
    
    for t in range(len(traffic)):
        if valid_mask[t]:
            tot = traffic[t].sum()
            total_demand.append(tot)
            # Rough hour approximation since we just need hour of day
            # start_dt + 15 mins * t
            # For exact we could use timedelta, but since it's just for plot:
            hr = (start_dt.hour + (t * 15) // 60) % 24
            hourly[hr].append(tot)
        else:
            total_demand.append(np.nan)

    plt.figure(figsize=(12, 3))
    plt.plot(total_demand, linewidth=0.5)
    plt.title("Total Demand Over Time (GEANT, Mbps) [COMPUTED]")
    plt.xlabel("Timestep (15-min intervals)")
    plt.ylabel("Total demand (Mbps)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plots", "total_demand_over_time.png"), dpi=120)
    plt.close()

    hours = sorted(hourly.keys())
    hourly_means = [np.mean(hourly[h]) for h in hours]
    plt.figure(figsize=(8, 4))
    plt.plot(hours, hourly_means, marker="o")
    plt.title("Mean Total Demand by Hour of Day (GEANT) [COMPUTED]")
    plt.xlabel("Hour (UTC)")
    plt.ylabel("Mean demand (Mbps)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plots", "mean_demand_by_hour.png"), dpi=120)
    plt.close()

if __name__ == "__main__":
    main()
