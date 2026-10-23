"""
Phase 8 Milestone 1 — Ryu QoS Controller.

Minimal Ryu OpenFlow 1.3 controller for the diamond topology:

             s2
            /  \
           /    \
h1 -- s1          s4 -- h2
           \    /
            \  /
             s3

Responsibilities
----------------
1. Accept OF1.3 connections from all four OVS switches.
2. Perform ARP proxying so h1/h2 can resolve each other's MAC addresses
   without flooding (flood-based ARP would defeat path selection).
3. Expose a REST endpoint for explicit path selection:
       POST /phase8/path   {"path": ["s1", "s2", "s4"]}
4. Translate the logical switch path into concrete (datapath_id, in_port,
   out_port) tuples and push OFPFlowMod rules to every switch on the path.
5. Install reverse-direction rules so ICMP/ping works bidirectionally.

What this controller does NOT do
---------------------------------
- No GNN, no GA, no telemetry collection. Those belong to later milestones.
- No production-grade error handling or authentication.
- No LLDP-based delay measurement (out of scope for Milestone 1).

Design notes
------------
- Switch names ("s1"..."s4") are derived from the Mininet name that OVS
  reports in the HELLO message's datapath_id. Mininet maps "s1" -> dpid 1,
  "s2" -> dpid 2, etc.
- Port discovery: the controller learns switch<->switch port numbers from
  Packet-In events and ARP probes it issues at startup.  Until discovery
  completes, path installation is rejected with an error response.
- All flow entries use a single priority (FLOW_PRIORITY=200) and cookie 0xPH8
  so they can be bulk-deleted before switching paths.

Running
-------
Inside Ubuntu-24.04 WSL, with the ryu-venv activated:

    ryu-manager --ofp-tcp-listen-port 6633 \\
                --wsapi-port 8080 \\
                /mnt/c/Users/Admin/OneDrive/Desktop/DN_proj/phase8/controller/qos_controller.py

Then in a second terminal start the Mininet topology:

    sudo python3 /mnt/c/Users/Admin/OneDrive/Desktop/DN_proj/phase8/emulation/diamond_topo.py
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict

from ryu.app.wsgi import ControllerBase, WSGIApplication, route
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import (CONFIG_DISPATCHER, MAIN_DISPATCHER,
                                    set_ev_cls)
from ryu.lib.packet import arp, ethernet, icmp, ipv4, packet
from ryu.ofproto import ofproto_v1_3
from ryu.lib import hub
from webob import Response
import time
import sys
import os

# Add parent directory to path so we can import phase8
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from phase8.orchestrator.telemetry_buffer import TelemetryBuffer

LOG = logging.getLogger("phase8.controller")

# ── constants ────────────────────────────────────────────────────────────────
FLOW_PRIORITY = 200          # above the table-miss default-drop (0)
ARP_PRIORITY  = 210          # ARP rules sit above IPv4 rules
COOKIE        = 0x508        # "Phase 8" cookie — used to bulk-delete our rules
IDLE_TIMEOUT  = 0            # permanent (until explicitly deleted)
H1_IP, H1_MAC = "10.0.0.1", "00:00:00:00:00:01"
H2_IP, H2_MAC = "10.0.0.2", "00:00:00:00:00:02"

# Mininet assigns dpid = switch number: s1->1, s2->2, s3->3, s4->4
DPID_TO_NAME = {1: "s1", 2: "s2", 3: "s3", 4: "s4"}
NAME_TO_DPID = {v: k for k, v in DPID_TO_NAME.items()}

# Host attachment: which switch and port hosts h1/h2.
# s1-eth1 hosts h1; s4-eth3 hosts h2.
# These are fixed by the deterministic addLink order in diamond_topo.py.
HOST_ATTACHMENT = {
    H1_MAC: ("s1", 1),   # h1 is on s1 port 1
    H2_MAC: ("s4", 3),   # h2 is on s4 port 3
}


class Phase8Controller(app_manager.RyuApp):
    """Minimal OF1.3 controller for Phase 8 Milestone 1."""

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {"wsgi": WSGIApplication}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # datapath_id -> datapath object
        self.datapaths: dict[int, object] = {}

        # (dpid_a, dpid_b) -> port number on dpid_a that faces dpid_b
        # Populated by _learn_port(); pre-seeded from topology knowledge.
        self.inter_switch_ports: dict[tuple[int, int], int] = {}

        wsgi = kwargs["wsgi"]
        wsgi.register(Phase8RestAPI, {"controller_app": self})

        # Seed inter-switch port map from the deterministic addLink order in
        # diamond_topo.py. Port numbers on each switch:
        #   s1: eth1=h1, eth2=s2, eth3=s3
        #   s2: eth1=s1, eth2=s4
        #   s3: eth1=s1, eth2=s4
        #   s4: eth1=s2, eth2=s3, eth3=h2
        self._seed_port_map()

        # Telemetry
        self.telemetry_buffer = TelemetryBuffer(capacity_bps=100_000_000)
        self.monitor_thread = hub.spawn(self._monitor)

    def _seed_port_map(self):
        """Pre-seed the inter-switch port map from topology knowledge."""
        # (from_dpid, to_dpid) -> out_port on from_dpid
        m = self.inter_switch_ports
        m[(1, 2)] = 2  # s1 -> s2 via s1-eth2
        m[(1, 3)] = 3  # s1 -> s3 via s1-eth3
        m[(2, 1)] = 1  # s2 -> s1 via s2-eth1
        m[(2, 4)] = 2  # s2 -> s4 via s2-eth2
        m[(3, 1)] = 1  # s3 -> s1 via s3-eth1
        m[(3, 4)] = 2  # s3 -> s4 via s3-eth2
        m[(4, 2)] = 1  # s4 -> s2 via s4-eth1
        m[(4, 3)] = 2  # s4 -> s3 via s4-eth2
        LOG.info("Inter-switch port map seeded (%d entries).", len(m))

    def _monitor(self):
        """Periodically poll connected switches for port statistics."""
        while True:
            for dp in self.datapaths.values():
                self._request_port_stats(dp)
            hub.sleep(1)

    def _request_port_stats(self, datapath):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        body = ev.msg.body
        dpid = ev.msg.datapath.id
        now = time.time()
        
        # Reverse map to find dst_switch for (dpid, port)
        # self.inter_switch_ports maps (src_dpid, dst_dpid) -> out_port
        port_to_dst = {}
        for (src, dst), port in self.inter_switch_ports.items():
            if src == dpid:
                port_to_dst[port] = DPID_TO_NAME.get(dst, f"s{dst}")
                
        src_name = DPID_TO_NAME.get(dpid, f"s{dpid}")

        for stat in body:
            port = stat.port_no
            if port in port_to_dst:
                dst_name = port_to_dst[port]
                # We only process telemetry for switch-to-switch links
                self.telemetry_buffer.process_port_stats(
                    dpid=dpid, port=port, timestamp=now,
                    rx_bytes=stat.rx_bytes, tx_bytes=stat.tx_bytes,
                    rx_packets=stat.rx_packets, tx_packets=stat.tx_packets,
                    rx_dropped=stat.rx_dropped, tx_dropped=stat.tx_dropped,
                    rx_errors=stat.rx_errors, tx_errors=stat.tx_errors,
                    src_node=src_name, dst_node=dst_name
                )

    # ── OF event handlers ────────────────────────────────────────────────────

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Install a table-miss rule that sends unknown packets to controller."""
        dp = ev.msg.datapath
        ofp, parser = dp.ofproto, dp.ofproto_parser
        self.datapaths[dp.id] = dp
        name = DPID_TO_NAME.get(dp.id, f"dpid={dp.id:#x}")
        LOG.info("Switch connected: %s (dpid=%d)", name, dp.id)

        # Table-miss: send to controller, no buffer.
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)]
        self._add_flow(dp, priority=0, match=match, actions=actions, cookie=0)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """Handle ARP Packet-In to provide MAC resolution without flooding."""
        msg = ev.msg
        dp = msg.datapath
        ofp, parser = dp.ofproto, dp.ofproto_parser
        in_port = msg.match["in_port"]

        pkt = packet.Packet(msg.data)
        eth_pkt = pkt.get_protocol(ethernet.ethernet)
        if eth_pkt is None:
            return

        arp_pkt = pkt.get_protocol(arp.arp)
        if arp_pkt is None:
            return  # only handle ARP at this milestone

        if arp_pkt.opcode != arp.ARP_REQUEST:
            return

        # Determine the MAC to reply with.
        target_ip = arp_pkt.dst_ip
        reply_mac = None
        if target_ip == H1_IP:
            reply_mac = H1_MAC
        elif target_ip == H2_IP:
            reply_mac = H2_MAC
        if reply_mac is None:
            return

        LOG.info("ARP request for %s on switch %s port %d — replying %s",
                 target_ip, DPID_TO_NAME.get(dp.id, dp.id), in_port, reply_mac)

        # Craft ARP reply.
        reply_pkt = packet.Packet()
        reply_pkt.add_protocol(ethernet.ethernet(
            ethertype=0x0806,
            dst=eth_pkt.src,
            src=reply_mac))
        reply_pkt.add_protocol(arp.arp(
            opcode=arp.ARP_REPLY,
            src_mac=reply_mac,
            src_ip=target_ip,
            dst_mac=arp_pkt.src_mac,
            dst_ip=arp_pkt.src_ip))
        reply_pkt.serialize()

        out = parser.OFPPacketOut(
            datapath=dp,
            buffer_id=ofp.OFP_NO_BUFFER,
            in_port=ofp.OFPP_CONTROLLER,
            actions=[parser.OFPActionOutput(in_port)],
            data=reply_pkt.data)
        dp.send_msg(out)

    # ── path installation ─────────────────────────────────────────────────────

    def install_path(self, switch_path: list[str]) -> dict:
        """Install OpenFlow rules for a logical switch path.

        Parameters
        ----------
        switch_path : list[str]
            Ordered list of switch names, e.g. ["s1", "s2", "s4"].
            Must start at s1 and end at s4.

        Returns
        -------
        dict with keys "status", "path", "rules_installed".
        """
        if not switch_path or switch_path[0] != "s1" or switch_path[-1] != "s4":
            return {"status": "error",
                    "message": "Path must start at s1 and end at s4."}

        dpids = []
        for name in switch_path:
            d = NAME_TO_DPID.get(name)
            if d is None:
                return {"status": "error", "message": f"Unknown switch: {name}"}
            if d not in self.datapaths:
                return {"status": "error",
                        "message": f"Switch {name} not yet connected to controller."}
            dpids.append(d)

        # Delete existing path rules from every switch so stale rules don't
        # interfere with the new path.
        self._delete_path_flows()

        rules_installed = []

        # ── forward direction: h1 -> h2 ──────────────────────────────────────
        for i, dpid in enumerate(dpids):
            dp = self.datapaths[dpid]
            parser = dp.ofproto_parser

            # Determine input and output ports for this switch.
            if i == 0:
                # First switch: packet arrives from h1.
                in_port = HOST_ATTACHMENT[H1_MAC][1]
            else:
                in_port = self.inter_switch_ports[(dpid, dpids[i - 1])]

            if i == len(dpids) - 1:
                # Last switch: forward toward h2.
                out_port = HOST_ATTACHMENT[H2_MAC][1]
            else:
                out_port = self.inter_switch_ports[(dpid, dpids[i + 1])]

            match = parser.OFPMatch(
                eth_type=0x0800,
                ipv4_src=H1_IP,
                ipv4_dst=H2_IP,
            )
            actions = [parser.OFPActionOutput(out_port)]
            self._add_flow(dp, FLOW_PRIORITY, match, actions, COOKIE)
            rules_installed.append(
                {"switch": DPID_TO_NAME[dpid], "direction": "h1->h2",
                 "in_port": in_port, "out_port": out_port})
            LOG.info("Installed h1->h2 rule on %s: out_port=%d",
                     DPID_TO_NAME[dpid], out_port)

        # ── reverse direction: h2 -> h1 (same physical path, reversed) ───────
        for i, dpid in enumerate(reversed(dpids)):
            dp = self.datapaths[dpid]
            parser = dp.ofproto_parser
            rev_i = len(dpids) - 1 - i

            if i == 0:
                # First switch in reverse = last in forward = s4.
                in_port = HOST_ATTACHMENT[H2_MAC][1]
            else:
                prev_dpid = dpids[rev_i + 1]
                in_port = self.inter_switch_ports[(dpid, prev_dpid)]

            if i == len(dpids) - 1:
                out_port = HOST_ATTACHMENT[H1_MAC][1]
            else:
                next_dpid = dpids[rev_i - 1]
                out_port = self.inter_switch_ports[(dpid, next_dpid)]

            match = parser.OFPMatch(
                eth_type=0x0800,
                ipv4_src=H2_IP,
                ipv4_dst=H1_IP,
            )
            actions = [parser.OFPActionOutput(out_port)]
            self._add_flow(dp, FLOW_PRIORITY, match, actions, COOKIE)
            rules_installed.append(
                {"switch": DPID_TO_NAME[dpid], "direction": "h2->h1",
                 "in_port": in_port, "out_port": out_port})
            LOG.info("Installed h2->h1 rule on %s: out_port=%d",
                     DPID_TO_NAME[dpid], out_port)

        return {"status": "ok", "path": switch_path,
                "rules_installed": rules_installed}

    # ── helpers ───────────────────────────────────────────────────────────────

    def _add_flow(self, dp, priority, match, actions, cookie=0,
                  idle_timeout=IDLE_TIMEOUT, hard_timeout=0):
        ofp, parser = dp.ofproto, dp.ofproto_parser
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(
            datapath=dp,
            cookie=cookie,
            priority=priority,
            match=match,
            instructions=inst,
            idle_timeout=idle_timeout,
            hard_timeout=hard_timeout,
        )
        dp.send_msg(mod)

    def _delete_path_flows(self):
        """Remove all flows previously installed by this controller (cookie match)."""
        for dp in self.datapaths.values():
            ofp, parser = dp.ofproto, dp.ofproto_parser
            mod = parser.OFPFlowMod(
                datapath=dp,
                cookie=COOKIE,
                cookie_mask=0xFFFF,
                command=ofp.OFPFC_DELETE,
                out_port=ofp.OFPP_ANY,
                out_group=ofp.OFPG_ANY,
                match=parser.OFPMatch(),
            )
            dp.send_msg(mod)
        LOG.info("Deleted all cookie=0x%X flows from %d switches.",
                 COOKIE, len(self.datapaths))

    def get_topology_info(self) -> dict:
        return {
            "connected_switches": [
                {"name": DPID_TO_NAME.get(did, f"dpid={did:#x}"), "dpid": did}
                for did in sorted(self.datapaths)
            ],
            "inter_switch_ports": {
                f"{DPID_TO_NAME.get(a,'?')}->{DPID_TO_NAME.get(b,'?')}": p
                for (a, b), p in sorted(self.inter_switch_ports.items())
            },
        }


# ── REST API ─────────────────────────────────────────────────────────────────

class Phase8RestAPI(ControllerBase):
    """REST endpoints for Milestone 1 path control."""

    def __init__(self, req, link, data, **cfg):
        super().__init__(req, link, data, **cfg)
        self.app: Phase8Controller = data["controller_app"]

    @route("phase8", "/phase8/path", methods=["POST"])
    def select_path(self, req, **_kwargs):
        """
        Install a forwarding path.

        Body: {"path": ["s1", "s2", "s4"]}   (switch names only, no hosts)
        """
        try:
            body = json.loads(req.body)
            path = body.get("path", [])
        except (ValueError, KeyError):
            return Response(status=400,
                            content_type="application/json",
                            body=json.dumps({"status": "error",
                                             "message": "Invalid JSON body."}))

        result = self.app.install_path(path)
        status = 200 if result.get("status") == "ok" else 400
        return Response(status=status,
                        content_type="application/json",
                        charset="UTF-8",
                        body=json.dumps(result, indent=2).encode("utf-8"))

    @route("phase8", "/phase8/topology", methods=["GET"])
    def topology_info(self, req, **_kwargs):
        """Return connected switches and inter-switch port map."""
        info = self.app.get_topology_info()
        return Response(status=200,
                        content_type="application/json",
                        charset="UTF-8",
                        body=json.dumps(info, indent=2).encode("utf-8"))

    @route("phase8", "/phase8/telemetry", methods=["GET"])
    def get_telemetry(self, req, **_kwargs):
        """Return the latest telemetry for all directed links."""
        telemetry = self.app.telemetry_buffer.get_latest_telemetry()
        return Response(status=200,
                        content_type="application/json",
                        charset="UTF-8",
                        body=json.dumps(telemetry, indent=2).encode("utf-8"))
    @route("phase8", "/phase8/health", methods=["GET"])
    def health(self, req, **_kwargs):
        return Response(status=200,
                        content_type="application/json",
                        charset="UTF-8",
                        body=json.dumps({"status": "ok"}).encode("utf-8"))
