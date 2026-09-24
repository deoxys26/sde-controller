import json

with open("data/experiments/phase7_model_d/results.json", "r") as f:
    data = json.load(f)

print("Protocol Configuration:")
for k, v in data["protocol"].items():
    print(f"  {k}: {v}")
print("\n--- AGGREGATE RESULTS ACROSS ALL SCENARIOS ---")
for method, metrics in data["aggregate_across_scenarios"].items():
    print(f"\n{method}:")
    print(f"  Delay:       {metrics['mean_link_delay']['mean']:.4f} ± {metrics['mean_link_delay']['std']:.4f}")
    print(f"  Utilization: {metrics['mean_link_utilization']['mean']:.4f} ± {metrics['mean_link_utilization']['std']:.4f}")
    print(f"  Packet Loss: {metrics['mean_link_packet_loss']['mean']:.4f} ± {metrics['mean_link_packet_loss']['std']:.4f}")
    print(f"  Jitter:      {metrics['mean_link_jitter']['mean']:.4f} ± {metrics['mean_link_jitter']['std']:.4f}")
    print(f"  Throughput:  {metrics['mean_link_throughput']['mean']:.4f} ± {metrics['mean_link_throughput']['std']:.4f}")
    print(f"  Failure Rate:{metrics['failure_rate']['mean']:.4f} ± {metrics['failure_rate']['std']:.4f}")
    print(f"  Samples:     {metrics['link_samples']} links, {metrics['flow_samples']} flows (failed {metrics['failed_flows']})")

print("\n--- SCENARIO-LEVEL COMPARISON (Model D vs Current) ---")
for scenario in data["scenario_results"]:
    name = scenario["scenario"]
    print(f"\nScenario: {name}")
    c = scenario["methods"]["predicted_qos_ga_current"]
    d = scenario["methods"]["predicted_qos_ga_model_d"]
    
    # Focus on key metrics
    print(f"  Delay:       C={c['mean_link_delay']:.4f} | D={d['mean_link_delay']:.4f}  " + ("(D BETTER)" if d['mean_link_delay'] < c['mean_link_delay'] else "(D WORSE)"))
    print(f"  Utilization: C={c['mean_link_utilization']:.4f} | D={d['mean_link_utilization']:.4f}  " + ("(D BETTER)" if d['mean_link_utilization'] < c['mean_link_utilization'] else "(D WORSE)"))
    print(f"  Packet Loss: C={c['mean_link_packet_loss']:.4f} | D={d['mean_link_packet_loss']:.4f}  " + ("(D BETTER)" if d['mean_link_packet_loss'] < c['mean_link_packet_loss'] else "(D WORSE)"))
