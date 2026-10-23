from phase8.emulation.diamond_topo import diamond_topology
import time, urllib.request, json
from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from phase8.emulation.diamond_topo import DiamondTopo

net = Mininet(topo=DiamondTopo(), controller=RemoteController, switch=OVSKernelSwitch)
net.start()
time.sleep(2)

def set_path(path_nodes):
    req = urllib.request.Request(
        "http://localhost:8080/phase8/path",
        data=json.dumps({"path": path_nodes}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        print(f"Set path response: {resp.status}")

set_path(["s1", "s2", "s4"])
time.sleep(1)

h1, h2 = net.get('h1', 'h2')
print("--- h1 ping -c 5 h2 ---")
print(h1.cmd('ping -c 5 h2'))

net.stop()
