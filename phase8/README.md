# Phase 8 — SDN Emulation

## Overview

Phase 8 moves from pure Python simulation (Phases 1–7) toward real SDN emulation using
Mininet + Open vSwitch + Ryu. Milestone 1 proves that the controller can
deliberately install and switch forwarding paths before any GNN or GA integration.

---

## Environment

| Component      | Version           | Location                     |
|----------------|-------------------|------------------------------|
| WSL Distribution | Ubuntu-24.04    | `wsl -d Ubuntu-24.04`        |
| Mininet        | 2.3.0             | System install in Ubuntu     |
| Open vSwitch   | 3.3.9             | System install in Ubuntu     |
| Ryu            | 4.34              | `~/ryu-venv` (Python 3.9 venv) |
| Python (Ryu)   | 3.9.25            | `/usr/bin/python3.9`         |
| Python (ML)    | 3.12.3 (Windows)  | Unchanged — Phase 1–7 env    |

> **Important**: Always use `wsl -d Ubuntu-24.04` — NOT the default `docker-desktop` distro.

---

## Milestone 1 — Diamond Topology + Manual Path Selection

### Topology

```
             s2
            /  \
           /    \
h1 -- s1          s4 -- h2
           \    /
            \  /
             s3
```

| Host | IP          | MAC               |
|------|-------------|-------------------|
| h1   | 10.0.0.1/24 | 00:00:00:00:00:01 |
| h2   | 10.0.0.2/24 | 00:00:00:00:00:02 |

**Path A**: `h1 → s1 → s2 → s4 → h2`  
**Path B**: `h1 → s1 → s3 → s4 → h2`

### OpenFlow Port Map (deterministic — from `addLink` order in `diamond_topo.py`)

| Switch | Port | Connected to |
|--------|------|-------------|
| s1     | 1    | h1          |
| s1     | 2    | s2          |
| s1     | 3    | s3          |
| s2     | 1    | s1          |
| s2     | 2    | s4          |
| s3     | 1    | s1          |
| s3     | 2    | s4          |
| s4     | 1    | s2          |
| s4     | 2    | s3          |
| s4     | 3    | h2          |

---

## Step-by-step: Starting Milestone 1

### Terminal 1 — Start Ryu controller

```bash
wsl -d Ubuntu-24.04
source ~/ryu-venv/bin/activate
ryu-manager --ofp-tcp-listen-port 6633 \
            --wsapi-port 8080 \
            /mnt/c/Users/Admin/OneDrive/Desktop/DN_proj/phase8/controller/qos_controller.py
```

You should see:
```
loading app phase8/controller/qos_controller.py
instantiating app ryu.controller.ofp_handler
...
```

### Terminal 2 — Start Mininet diamond topology

```bash
wsl -d Ubuntu-24.04
cd /mnt/c/Users/Admin/OneDrive/Desktop/DN_proj
sudo python3 phase8/emulation/diamond_topo.py
```

Ryu terminal should then show 4 switch connections:
```
Switch connected: s1 (dpid=1)
Switch connected: s2 (dpid=2)
Switch connected: s3 (dpid=3)
Switch connected: s4 (dpid=4)
```

---

## Selecting a Path

### Verify controller health

```bash
curl http://localhost:8080/phase8/health
```

### Check connected switches

```bash
curl http://localhost:8080/phase8/topology
```

### Install Path A (h1 → s1 → s2 → s4 → h2)

```bash
curl -X POST http://localhost:8080/phase8/path \
     -H "Content-Type: application/json" \
     -d '{"path": ["s1","s2","s4"]}'
```

### Install Path B (h1 → s1 → s3 → s4 → h2)

```bash
curl -X POST http://localhost:8080/phase8/path \
     -H "Content-Type: application/json" \
     -d '{"path": ["s1","s3","s4"]}'
```

You can switch between A and B at any time without restarting Mininet.

---

## Verifying Path Usage

### 1. Ping test (basic connectivity)

In the Mininet CLI:
```
mininet> h1 ping -c 4 h2
```

### 2. OVS flow counters (proves which switch forwarded packets)

```bash
# Path A active: s2 should show h1->h2 rules; s3 should not
sudo ovs-ofctl -O OpenFlow13 dump-flows s2 | grep "nw_src=10.0.0.1"
sudo ovs-ofctl -O OpenFlow13 dump-flows s3 | grep "nw_src=10.0.0.1"
```

### 3. Automated verification script

```bash
# After installing Path A and running ping:
wsl -d Ubuntu-24.04 python3 \
    /mnt/c/Users/Admin/OneDrive/Desktop/DN_proj/phase8/tests/verify_path.py A

# After switching to Path B:
wsl -d Ubuntu-24.04 python3 \
    /mnt/c/Users/Admin/OneDrive/Desktop/DN_proj/phase8/tests/verify_path.py B
```

Expected output for Path A:
```
Switch s2:
  h1->h2 flow rules: 1
  h1->h2 packet count: 4
Switch s3:
  h1->h2 flow rules: 0
  ...
Active path: A (via s2)
PASS: Expected path A is active.
```

---

## Limitations at Milestone 1

- No GNN forecasting, no GA routing. Path is selected manually via REST API.
- No delay/jitter telemetry. OpenFlow port counters provide bytes/packets only.
- Security (UNSW-NB15) is not connected. The security interface remains a separate
  component as documented in the Phase 5/7 findings.
- No statistical claims about routing performance.

---

## File Structure

```
phase8/
├── controller/
│   └── qos_controller.py    # Ryu OF1.3 controller with REST path API
├── emulation/
│   └── diamond_topo.py      # Mininet diamond topology
├── tests/
│   └── verify_path.py       # OVS flow counter verification script
└── README.md                # This file
```

---

## WSL-Specific Notes

- Always use `wsl -d Ubuntu-24.04` — the default WSL distro is `docker-desktop` which has no Mininet.
- Ryu runs in an isolated Python 3.9 venv at `~/ryu-venv` in Ubuntu-24.04.
- The ML environment (PyTorch, Phase 1–7) remains in Windows Python 3.12 — untouched.
- Mininet requires `sudo` for network namespace operations.
- The Ryu REST API listens on `localhost:8080` within the WSL network namespace.

---

## Milestone 2 (future)

After Milestone 1 is fully verified, the next step is telemetry collection:
- OVS port statistics polling (bytes/packets/drops per interface per interval).
- A scientifically justified approach to delay measurement (active probe packets).
- Adapting the collected telemetry into the feature format expected by Model B.
