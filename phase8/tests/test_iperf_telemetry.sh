#!/bin/bash
set -e

sudo mn -c > /dev/null 2>&1 || true

source /home/phani26/ryu-venv/bin/activate
ryu-manager --ofp-tcp-listen-port 6633 --wsapi-port 8080 /mnt/c/Users/Admin/OneDrive/Desktop/DN_proj/phase8/controller/qos_controller.py > ryu_iperf.log 2>&1 &
RYU_PID=$!

sleep 3

sudo python3 /mnt/c/Users/Admin/OneDrive/Desktop/DN_proj/phase8/tests/test_iperf_telemetry.py

kill $RYU_PID
