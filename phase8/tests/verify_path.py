#!/usr/bin/env python3
"""
Phase 8 Milestone 1 — Path verification script.

Queries OVS flow counters to confirm which physical path h1->h2 traffic
is actually traversing, without relying only on a successful ping.

Usage (inside Ubuntu-24.04 WSL):
    python3 /mnt/c/Users/Admin/OneDrive/Desktop/DN_proj/phase8/tests/verify_path.py [A|B]

The script:
1. Dumps flow counters from s2 and s3.
2. Checks which switch has IPv4 h1->h2 forwarding rules installed.
3. Reports whether the expected path (A via s2 or B via s3) is active.
"""

import subprocess
import sys
import re


def ovs_dump_flows(switch_name: str) -> str:
    """Run ovs-ofctl dump-flows on the given switch and return output."""
    result = subprocess.run(
        ["sudo", "ovs-ofctl", "-O", "OpenFlow13", "dump-flows", switch_name],
        capture_output=True, text=True
    )
    return result.stdout + result.stderr


def count_h1h2_flows(flow_output: str) -> int:
    """Count h1->h2 IPv4 forwarding rules in ovs-ofctl output."""
    count = 0
    for line in flow_output.splitlines():
        if "nw_src=10.0.0.1" in line and "nw_dst=10.0.0.2" in line:
            count += 1
    return count


def get_flow_packet_count(flow_output: str, src_ip: str, dst_ip: str) -> int:
    """Extract packet count for a specific src->dst flow."""
    total = 0
    for line in flow_output.splitlines():
        if f"nw_src={src_ip}" in line and f"nw_dst={dst_ip}" in line:
            m = re.search(r"n_packets=(\d+)", line)
            if m:
                total += int(m.group(1))
    return total


def main():
    expected = sys.argv[1].upper() if len(sys.argv) > 1 else None

    print("=== Phase 8 Milestone 1: Path Verification ===\n")

    for sw in ["s2", "s3", "s4"]:
        flows = ovs_dump_flows(sw)
        h1h2_count = count_h1h2_flows(flows)
        pkt_count = get_flow_packet_count(flows, "10.0.0.1", "10.0.0.2")
        print(f"Switch {sw}:")
        print(f"  h1->h2 flow rules: {h1h2_count}")
        print(f"  h1->h2 packet count: {pkt_count}")
        print()

    # Determine which path is active based on flow rule presence
    s2_flows = ovs_dump_flows("s2")
    s3_flows = ovs_dump_flows("s3")

    s2_has_rule = count_h1h2_flows(s2_flows) > 0
    s3_has_rule = count_h1h2_flows(s3_flows) > 0

    if s2_has_rule and not s3_has_rule:
        active = "A (via s2)"
    elif s3_has_rule and not s2_has_rule:
        active = "B (via s3)"
    elif s2_has_rule and s3_has_rule:
        active = "BOTH (unexpected — clear flow tables)"
    else:
        active = "NONE (no h1->h2 rules installed yet)"

    print(f"Active path: {active}")

    if expected:
        path_char = "A" if "A" in active else ("B" if "B" in active else "?")
        if path_char == expected:
            print(f"PASS: Expected path {expected} is active.")
        else:
            print(f"FAIL: Expected path {expected} but got {active}.")
            sys.exit(1)


if __name__ == "__main__":
    main()
