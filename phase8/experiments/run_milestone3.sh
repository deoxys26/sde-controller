#!/bin/bash
# Phase 8 Milestone 3 – run the frozen Model B inference experiment.
# Must be executed inside Ubuntu-24.04 WSL as root.
set -e

sudo mn -c > /dev/null 2>&1 || true

source /home/phani26/ryu-venv/bin/activate
ryu-manager --ofp-tcp-listen-port 6633 --wsapi-port 8080 \
    /mnt/c/Users/Admin/OneDrive/Desktop/DN_proj/phase8/controller/qos_controller.py \
    > ryu_m3.log 2>&1 &
RYU_PID=$!

sleep 4

# Use the venv python which has numpy, torch, ryu
sudo -E /home/phani26/ryu-venv/bin/python3 \
    /mnt/c/Users/Admin/OneDrive/Desktop/DN_proj/phase8/experiments/run_model_b_experiment.py

kill $RYU_PID 2>/dev/null || true
