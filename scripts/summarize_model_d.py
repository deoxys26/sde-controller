import json
import numpy as np
from collections import defaultdict

with open("data/experiments/model_d/results.json", "r") as f:
    data = json.load(f)

protocol = data["protocol"]
records = data["records"]

print(f"Total configurations: {len(records)}")
print("Protocol:", protocol)

scenarios = protocol["scenarios"]
models = protocol["models"]
seeds = protocol["seeds"]

# Group by scenario and model
# stats[scenario][model][metric] = list of seed values
stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
params = {}

for r in records:
    scenario = r["scenario"]
    model = r["model"]
    seed = r["training_seed"]
    metrics = r["metrics"]
    params[model] = r["parameter_count"]
    
    stats[scenario][model]["delay"].append(metrics["delay"]["mae"])
    stats[scenario][model]["utilization"].append(metrics["utilization"]["mae"])
    stats[scenario][model]["packet_loss"].append(metrics["packet_loss"]["mae"])
    stats[scenario][model]["jitter"].append(metrics["jitter"]["mae"])
    stats[scenario][model]["aggregate"].append(metrics["aggregate"]["mae"])

print("\nParams:", params)

print("\n--- Mean across 3 seeds ---")
for scenario in scenarios:
    print(f"\nScenario: {scenario}")
    for model in models:
        aggs = stats[scenario][model]["aggregate"]
        dels = stats[scenario][model]["delay"]
        utils = stats[scenario][model]["utilization"]
        ploss = stats[scenario][model]["packet_loss"]
        jits = stats[scenario][model]["jitter"]
        
        print(f"  {model:8s} | Agg: {np.mean(aggs):.5f} ± {np.std(aggs):.5f} | "
              f"D: {np.mean(dels):.5f} | U: {np.mean(utils):.5f} | "
              f"L: {np.mean(ploss):.5f} | J: {np.mean(jits):.5f}")

# Overall comparison
print("\n--- Overall Aggregate MAE ---")
for model in models:
    all_aggs = []
    for scenario in scenarios:
        all_aggs.extend(stats[scenario][model]["aggregate"])
    print(f"{model:8s}: {np.mean(all_aggs):.5f}")

print("\n--- Target-level Wins (Model D vs B/C) ---")
targets = ["delay", "utilization", "packet_loss", "jitter", "aggregate"]
for target in targets:
    win_b = 0
    win_c = 0
    lose_b = 0
    lose_c = 0
    for scenario in scenarios:
        mean_b = np.mean(stats[scenario]["current"][target])
        mean_c = np.mean(stats[scenario]["proposed"][target])
        mean_d = np.mean(stats[scenario]["model_d"][target])
        
        if mean_d < mean_b: win_b += 1
        elif mean_d > mean_b: lose_b += 1
        
        if mean_d < mean_c: win_c += 1
        elif mean_d > mean_c: lose_c += 1
        
    print(f"{target.upper()}:")
    print(f"  vs B (current)  -> Better in {win_b}, Worse in {lose_b}")
    print(f"  vs C (proposed) -> Better in {win_c}, Worse in {lose_c}")
