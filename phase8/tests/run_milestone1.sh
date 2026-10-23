#!/bin/bash
set -e

echo "Cleaning mininet..."
sudo mn -c > /dev/null 2>&1 || true

echo "Starting Ryu Controller in background..."
source /home/phani26/ryu-venv/bin/activate
ryu-manager --ofp-tcp-listen-port 6633 --wsapi-port 8080 /mnt/c/Users/Admin/OneDrive/Desktop/DN_proj/phase8/controller/qos_controller.py > ryu.log 2>&1 &
RYU_PID=$!

echo "Waiting for Ryu to start..."
sleep 3

echo "Starting Mininet automated tests..."
sudo python3 /mnt/c/Users/Admin/OneDrive/Desktop/DN_proj/phase8/tests/test_diamond_topo.py

echo "Killing Ryu..."
kill $RYU_PID
