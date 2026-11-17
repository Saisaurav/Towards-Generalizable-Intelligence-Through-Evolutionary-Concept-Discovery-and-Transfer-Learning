# FrameForge Robotics Lab

The robotics tab adds two built-in tasks to the existing FrameForge project:

1. **TurtleBot navigation** — differential-drive navigation with 24-ray simulated LiDAR.
2. **2-link arm reaching** — planar two-joint manipulation with a filtered joint estimate.

## What works in this integration

- Both environments are registered in the same C++ NEAT + PPO learner as the game tasks.
- The included trained policies are in `policies/` and are loaded by the browser UI.
- The frontend can launch fresh training jobs, poll generation progress, show logs and load the resulting policy.
- `web/robotics_sim.js` evaluates the exported NEAT topology and runs a matching browser-side surrogate preview.
- Every training job records population snapshots under `robotics_sim/runs/<job-id>/genomes/`.
- `ENV_SPECS.md` documents observation/action layouts and the arm's `feat=3` filter.

## Demo flow

1. Start the normal FrameForge server.
2. Open **Robot Lab** in the left navigation.
3. Choose TurtleBot or 2-Link Arm.
4. Click **Load saved demo policy** for an immediate demo, or choose a step budget and click **Train brain**.
5. Use Pause/Run and Reset to control the browser-side policy preview.

The upload panel is deliberately a placeholder. Selecting a file only displays its name; custom URDF/SDF parsing is not implemented. The browser preview is a surrogate visualization, not a live Gazebo renderer. The backend does not launch Gazebo.

Quick policy checks (from the project root):

```bash
node robotics_sim/test_policy_preview.js
python robotics_sim/ros2/test_policy_runtime.py
```

## Build the robotics brain

The server builds `src/brain_robotics` automatically when a C++17 compiler is available. To build manually:

```bash
cd src
make brain
```

The updated `main.cpp` accepts:

```bash
./brain turtlebot ../robotics_sim/none.txt 25000000 3 48 ../robotics_sim/policies/turtlebot.json \
  --preset lam --env-args "rnoise=0.05,rdelay=2,rlag=1.5" \
  --genome-dir ../robotics_sim/runs/turtlebot/genomes

./brain arm2link ../robotics_sim/none.txt 8000000 3 48 ../robotics_sim/policies/arm2link.json \
  --preset lam --env-args "rnoise=0.05,rdelay=2,rlag=1.5,feat=3,fk=0.25" \
  --genome-dir ../robotics_sim/runs/arm2link/genomes
```

## ROS 2 / Gazebo bridge scripts

`ros2/` contains minimal ROS 2 nodes for deployment against compatible Gazebo models. They load the exported policy JSON and translate sensor observations to velocity commands. They are generic adapters, not turnkey robot descriptions: your Gazebo model/controllers must expose the configured topics and joint names. Review limits and test in simulation with conservative velocity limits before connecting hardware. The included policies have only been evaluated in the C++ surrogate, not in Gazebo or on a physical robot.
