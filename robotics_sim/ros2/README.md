# ROS 2 policy bridges

These are minimal adapters for a compatible Gazebo/ROS 2 setup. They load the actual exported NEAT policies from `robotics_sim/policies/` and reproduce the documented observation order.

## TurtleBot

Expected topics:

- `/scan` (`sensor_msgs/msg/LaserScan`)
- `/odom` (`nav_msgs/msg/Odometry`)
- `/cmd_vel` (`geometry_msgs/msg/Twist`)

Run from a ROS 2 environment with `rclpy`, `sensor_msgs`, `nav_msgs` and `geometry_msgs` installed:

```bash
python3 robotics_sim/ros2/turtlebot_policy_node.py --ros-args \
  -p goal_x:=1.0 -p goal_y:=1.0
```

The goal is expressed in the odometry frame. Configure topic names and velocity caps to match the robot model.

## 2-link arm

Expected topics:

- `/joint_states` (`sensor_msgs/msg/JointState`)
- `/joint_group_velocity_controller/commands` (`std_msgs/msg/Float64MultiArray`)

Run with joint names matching your URDF/controller:

```bash
python3 robotics_sim/ros2/arm2link_policy_node.py --ros-args \
  -p joint_1_name:=joint1 -p joint_2_name:=joint2 \
  -p target_x:=0.38 -p target_y:=0.12
```

The arm bridge uses the trained complementary filter with gain 0.25 and publishes velocity commands in rad/s. Change the command topic to the one exposed by your velocity controller.

## Safety / compatibility

These nodes do not load or spawn a URDF/SDF, start Gazebo, perform collision checking, or certify sim-to-real transfer. Confirm joint names, controller interfaces, workspace, joint limits, emergency stop and velocity limits in a paused/isolated simulation first. The included policies were validated against the C++ surrogate only; no live Gazebo or physical-robot result is claimed.
