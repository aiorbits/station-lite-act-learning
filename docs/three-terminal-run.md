# Three-Terminal ACT Deployment Guide

This guide runs one ACT checkpoint on a prepared GPU computer and a prepared Station Lite robot computer. It assumes the Conda environments, robot SDK, calibration, CAN interfaces, cameras, and network connectivity are already configured. It does not explain how to configure those systems.

Keep datasets, checkpoints, credentials, calibration files, and run logs outside the Git repository. Use the GPU computer's current LAN address that is reachable from the robot; this address may change after reconnecting to the network.

## Terminal 1 — GPU computer: serve the policy

Run this on the GPU computer. Replace the repository and checkpoint paths with the actual paths on that computer. The checkpoint path must point to the pretrained_model directory.

~~~bash
conda activate station_act
cd /path/to/station-lite-act-learning

# Find the GPU computer's LAN address reachable from the robot.
hostname -I

# Select the trained checkpoint.
CHECKPOINT="$PWD/checkpoints/station_lite_act_baseline/checkpoints/020000/pretrained_model"

# Keep this terminal open while the robot client is running.
python deployment/station_act_http_server.py --checkpoint "$CHECKPOINT" --device cuda --host 0.0.0.0 --port 8008
~~~

The HTTP server loads the policy and exposes inference endpoints. It does not connect to robot hardware.

## Terminal 2 — robot computer: start the camera image service

Run this on the robot computer. Keep an already-running image service. Start it only if port 60000 is not listening. If the command stays in the foreground, leave this terminal open.

~~~bash
conda activate lumosteleop
cd /home/lumos/lumosteleop

# Inspect managed service state.
./lumosteleop status

# Start the camera service only when it is not already listening.
if ss -ltn | grep -q ':60000'; then echo 'Camera image service is already listening on port 60000.'; else ./lumosteleop image-server; fi
~~~

Before continuing, verify that ss -ltn | grep ':60000' reports a listener.

## Terminal 3 — robot computer: verify and run the client

Run this in a separate terminal on the robot computer. Replace GPU_LAN_IP with the current reachable LAN address printed in Terminal 1. Confirm that /health reports the intended checkpoint.

~~~bash
conda activate lumosteleop
cd /home/lumos/replay_work

# Replace this placeholder with the GPU computer's current reachable LAN IP.
GPU_HOST='GPU_LAN_IP'
SERVER="http://$GPU_HOST:8008"

# Check the policy server and camera service before starting inference.
curl -fsS --max-time 5 "$SERVER/health"
echo
ss -ltn | grep ':60000'

# Inference-only connectivity check; this does not move the robot.
# It sends a zero-filled state, so it does not validate live robot state.
python /path/to/station-lite-act-learning/deployment/station_lite_client.py --server "$SERVER" --loops 1
~~~

### First motion test: one short action prefix

Proceed only after the operator confirms the robot starting pose, a clear workspace, a clear return-home route, and an accessible emergency stop. The --execute flag enables motion. This client initializes both arms and returns both arms home when it exits.

~~~bash
python /path/to/station-lite-act-learning/deployment/station_lite_client.py --server "$SERVER" --execute --loops 1 --speed 0.2 --exec-steps 15 --switch-after 10 --blend-overlap 5 --margin 0.02
~~~

Observe the full test, including the automatic return-home motion. Continue only if movement is safe and expected.

### Longer pick-and-place run

Use this only after the short test has passed and an instructor has approved the run. Keep the operator at the emergency stop and record grasp and placement outcomes separately.

~~~bash
python /path/to/station-lite-act-learning/deployment/station_lite_client.py --server "$SERVER" --execute --loops 12 --speed 0.2 --exec-steps 30 --switch-after 20 --blend-overlap 10 --margin 0.02
~~~

Save logs locally with the checkpoint identifier, code revision, scene configuration, and outcome labels. A successful inference response or completed client process does not by itself establish a successful pick-and-place task.
