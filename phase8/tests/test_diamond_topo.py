#!/usr/bin/env python3
"""
Phase 8 Milestone 1 — Diamond Mininet topology.

Topology:
             s2
            /  \
           /    \
h1 -- s1          s4 -- h2
           \    /
            \  /
             s3

Two forwarding paths exist:
  Path A: h1 -> s1 -> s2 -> s4 -> h2
  Path B: h1 -> s1 -> s3 -> s4 -> h2

The active path is controlled exclusively by the Ryu controller via OpenFlow 1.3
rules; Mininet topology structure never changes between path selections.

Usage (inside Ubuntu-24.04 WSL, with Ryu already running):
    sudo python3 /mnt/c/Users/Admin/OneDrive/Desktop/DN_proj/phase8/emulation/diamond_topo.py
"""

from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel, info
from mininet.cli import CLI


def diamond_topology():
    """Create and return a Mininet network with the diamond topology."""
    net = Mininet(
        controller=RemoteController,
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=True,   # deterministic MAC assignment ordered by node name
        waitConnected=True,
    )

    info("*** Adding Ryu remote controller\n")
    c0 = net.addController(
        "c0",
        controller=RemoteController,
        ip="127.0.0.1",
        port=6633,
    )

    info("*** Adding switches (OpenFlow 1.3)\n")
    s1 = net.addSwitch("s1", cls=OVSSwitch, protocols="OpenFlow13")
    s2 = net.addSwitch("s2", cls=OVSSwitch, protocols="OpenFlow13")
    s3 = net.addSwitch("s3", cls=OVSSwitch, protocols="OpenFlow13")
    s4 = net.addSwitch("s4", cls=OVSSwitch, protocols="OpenFlow13")

    info("*** Adding hosts\n")
    h1 = net.addHost("h1", ip="10.0.0.1/24", mac="00:00:00:00:00:01")
    h2 = net.addHost("h2", ip="10.0.0.2/24", mac="00:00:00:00:00:02")

    info("*** Adding links\n")
    # Host links — port numbering is deterministic because Mininet assigns
    # interface indices in addLink call order.
    net.addLink(h1, s1, bw=100)   # h1-eth0  <->  s1-eth1

    # Path A backbone: s1 -> s2 -> s4
    net.addLink(s1, s2, bw=100)   # s1-eth2  <->  s2-eth1
    net.addLink(s2, s4, bw=100)   # s2-eth2  <->  s4-eth1

    # Path B backbone: s1 -> s3 -> s4
    net.addLink(s1, s3, bw=100)   # s1-eth3  <->  s3-eth1
    net.addLink(s3, s4, bw=100)   # s3-eth2  <->  s4-eth2

    net.addLink(h2, s4, bw=100)   # h2-eth0  <->  s4-eth3

    info("*** Starting network\n")
    net.build()
    c0.start()
    for sw in (s1, s2, s3, s4):
        sw.start([c0])

    info("*** Waiting for switches to connect to controller\n")
    net.waitConnected()

    info("\n*** Diamond topology ready.\n")
    info("    Path A: h1 -> s1 -> s2 -> s4 -> h2\n")
    info("    Path B: h1 -> s1 -> s3 -> s4 -> h2\n")
    info("\n    Select path via Ryu REST API:\n")
    info('      curl -X POST http://localhost:8080/phase8/path \\\n')
    info('           -H "Content-Type: application/json" \\\n')
    info("           -d '{\"path\": [\"s1\",\"s2\",\"s4\"]}'\n")
    info("\n*** Running automated tests for Milestone 1\n")
    import time, subprocess, urllib.request, json
    
    time.sleep(2)  # Wait for topology to settle

    def set_path(path_nodes):
        req = urllib.request.Request(
            "http://localhost:8080/phase8/path",
            data=json.dumps({"path": path_nodes}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req) as resp:
            info(f"Set path response: {resp.status}\n")

    def run_verify(expected):
        res = subprocess.run(["python3", "/mnt/c/Users/Admin/OneDrive/Desktop/DN_proj/phase8/tests/verify_path.py", expected], capture_output=True, text=True)
        info(res.stdout)
        if res.returncode != 0:
            info(f"Verification FAILED for path {expected}\n")
        return res.returncode

    # 1. Baseline ping (should fail or loop if no rules, but table-miss sends to controller)
    info("\n--- Baseline Ping ---\n")
    net.ping([h1, h2])

    # 2. Test Path A
    info("\n--- Testing Path A ---\n")
    set_path(["s1", "s2", "s4"])
    time.sleep(1)
    
    info("\n--- ping -c 5 ---\n")
    info(h1.cmd('ping -c 5 10.0.0.2'))

    errA = run_verify("A")

    # 3. Test Path B
    info("\n--- Testing Path B ---\n")
    set_path(["s1", "s3", "s4"])
    time.sleep(1)
    net.ping([h1, h2])
    errB = run_verify("B")

    info("\n*** Automated tests complete.\n")
    net.stop()
    
    if errA != 0 or errB != 0:
        import sys
        sys.exit(1)


if __name__ == "__main__":
    setLogLevel("info")
    diamond_topology()
