# Robotics environment specifications

These are fast C++ surrogate environments used to train NEAT + PPO policies. They are not Gazebo worlds and do not automatically parse uploaded URDF/SDF files.

## TurtleBot navigation (`turtlebot`)

- Model: TurtleBot3 Burger-like differential-drive robot in a 4 m × 4 m arena.
- Geometry: robot radius 0.105 m; circular obstacles; randomized start, goal and 3–6 obstacles.
- Observation: 24 range rays (3.5 m maximum range), normalized goal distance, sine/cosine of relative goal bearing, normalized linear/angular velocity, previous normalized action, normalized episode time.
- Observation count: 32.
- Action: two continuous values in `[-1, 1]`.
- Mapping: `linear.x = action[0] × 0.22 m/s`; `angular.z = action[1] × 2.0 rad/s`.
- Control period: 0.1 s (10 Hz).
- Dynamics: first-order actuator lag; optional command delay; observation noise; episode-wise domain randomization.
- Episode horizon: 300 steps.
- Success: reach within 0.15 m of the goal before collision or timeout.
- Training configuration used for the included policy: 25,000,000 environment steps, seed 3, population 48, `--preset lam`, `rnoise=0.05,rdelay=2,rlag=1.5`.

## Planar two-link arm (`arm2link`)

- Model: planar 2-DOF arm with link lengths 0.30 m and 0.25 m.
- Observation: `sin/cos(q1,q2)`, normalized joint velocities, end-effector target error, previous normalized action, fine-scale error, Jacobian-transpose error features, normalized episode time.
- Observation count: 15 for the included `feat=3` policy.
- Action: two continuous joint-velocity commands in `[-1, 1]`.
- Mapping: `joint_velocity[i] = action[i] × 2.0 rad/s`.
- Control period: 0.05 s (20 Hz).
- Dynamics: first-order actuator lag; optional command delay; joint-angle and velocity observation noise; episode-wise domain randomization.
- Episode horizon: 100 steps.
- Success: end-effector within 0.03 m of the target and joint speed below 0.1 (normalized) at the end of the episode.
- Included policy uses `feat=3,fk=0.25`: joint-angle complementary filter. Each observation uses `q_pred = qhat_prev + previous_action × 2.0 × 0.05`, then `qhat = q_pred + 0.25 × (measured_q - q_pred)`. The filtered angles feed sin/cos, forward-kinematics error and `Jᵀe` features.
- Training configuration used for the included policy: 8,000,000 environment steps, seed 3, population 48, `--preset lam`, `rnoise=0.05,rdelay=2,rlag=1.5,feat=3,fk=0.25`.

## Exported policy JSON

Each `robotics_sim/policies/*.json` contains:

- `env` and `env_args` identifying the exact training environment/configuration.
- `input_ids` in the network input order.
- `obs_names` documenting the observation vector order.
- `nodes` and `connections` describing the evolved NEAT topology and weights.
- `act_dim`, `logstd`, `action_scale`, and `rate_hz` for deployment.

The deterministic deployment action is the policy mean, clipped to `[-1, 1]`. The value output is not used for control.

## Included checkpoint results

| Environment | Steps | Seed | Held-out episodes | Success | Mean return |
|---|---:|---:|---:|---:|---:|
| TurtleBot | 25,000,000 | 3 | 300 | 86.3% | 3.1088 |
| 2-link arm | 8,000,000 | 3 | 300 | 89.0% | -2.9754 |

These are single-seed results on the C++ surrogate. They are not a multi-seed reliability claim and are not real-Gazebo or hardware validation.
