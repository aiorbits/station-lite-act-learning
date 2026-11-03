# Three-Terminal ACT Deployment Guide

This guide runs one ACT checkpoint across a prepared GPU computer and a prepared Station Lite robot computer. It assumes the required Conda environments, robot SDK, calibration, CAN interfaces, cameras, and network connectivity have already been configured. The repository does not configure those systems.

Keep the dataset, checkpoint, credentials, calibration files, and run logs outside the Git repository. Use the current GPU computer LAN address reachable from the robot; the address may change after reconnecting to the network.

## Terminal 1 — GPU computer: serve the policy

Run this on the GPU computer. Replace the repository and checkpoint paths with the actual paths on that computer. The checkpoint path must point to the pretrained_model directory.

~~~bash
conda activate station_act
cd /path/to/station-lite-act-learning

# Print the GPU computer addresses. Use the LAN address that the robot can reach.
hostname -I

# Set this to the selected ACT checkpoint directory.
CHECKPOINT="$PWD/checkpoints/station_lite_act_baseline/checkpoints/020000/pretrained_model"

# Keep this terminal open while the robot client is running.
python deployment/station_act_http_server.py \
  --checkpoint "$CHECKPOINT" \
  --device cuda --host 0.0.0.0 --port 8008
~~~

The server loads a policy and exposes HTTP endpoints. It does not connect to robot hardware.

## Terminal 2 — robot computer: start the camera image service

Run this on the robot computer. Leave an already-running image service alone. If port 60000 is not listening, start the service and keep it available while the client runs.

~~~bash
conda activate lumosteleop
cd /home/lumos/lumosteleop

# Review the managed service state.
./lumosteleop status

# Start the camera image service only when port 60000 is not already listening.
if ss -ltn | grep -q ':60000'; then
  echo "Camera image service is already listening on port 60000."
else
  ./lumosteleop image-server
fi
~~~

Before continuing, verify in another shell that ss -ltn | grep ':60000' reports a listener. If image-server runs in the foreground, leave Terminal 2 open.

## Terminal 3 — robot computer: verify and run the client

Run this in a separate terminal on the robot computer. Replace GPU_LAN_IP with the current reachable address printed in Terminal 1. Confirm that /health reports the checkpoint you intend to test.

~~~bash
conda activate lumosteleop
cd /path/to/station-lite-act-learning

# Replace this placeholder with the GPU computer current reachable LAN address.
GPU_HOST="GPU_LAN_IP"
SERVER="http://${GPU_HOST}:8008"

# Verify policy server and camera service before starting inference.
curl -fsS --max-time 5 "$SERVER/health"
echo
ss -ltn | grep ':60000'

# Inference-only connectivity check: no robot initialization or motion occurs.
# The client sends a zero-filled state in this mode, so this does not validate
# the robot live state or prove that a motion command is safe.
python deployment/station_lite_client.py --server "$SERVER" --loops 1
~~~

### First motion test: one short action prefix

Only continue after the operator confirms the robot starting pose, clear workspace, clear return-home route, and accessible emergency stop. The --execute flag enables motion. This client initializes both arms and returns both arms home when it exits.

~~~bash
python deployment/station_lite_client.py \
  --server "$SERVER" --execute \
  --loops 1 --speed 0.2 \
  --exec-steps 15 --switch-after 10 --blend-overlap 5 \
  --margin 0.02
~~~

Observe the complete short test, including the automatic return-home motion. Continue only if the motion is safe and expected.

### Longer pick-and-place run

Use this only after the short test has passed and an instructor has approved the run. Keep the operator at the emergency stop and record grasp and placement outcomes separately.

~~~bash
python deployment/station_lite_client.py \
  --server "$SERVER" --execute \
  --loops 12 --speed 0.2 \
  --exec-steps 30 --switch-after 20 --blend-overlap 10 \
  --margin 0.02
~~~

Save logs locally with the model checkpoint, code revision, scene configuration, and outcome labels. A successful inference response or completed client process does not by itself establish a successful pick-and-place task.
