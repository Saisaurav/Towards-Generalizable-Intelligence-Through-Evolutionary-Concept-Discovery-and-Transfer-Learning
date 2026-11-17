"""ROS 2 bridge for the exported TurtleBot surrogate policy.

Expected topics: sensor_msgs/LaserScan /scan, nav_msgs/Odometry /odom,
geometry_msgs/Twist /cmd_vel. Goal coordinates are expressed in odom frame.
"""
from __future__ import annotations
import math
import os
from pathlib import Path
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from neat_policy_runtime import NeatPolicy

class TurtleBotPolicyNode(Node):
    def __init__(self):
        super().__init__("frameforge_turtlebot_policy")
        default_policy = Path(__file__).resolve().parents[1] / "policies" / "turtlebot.json"
        self.declare_parameter("policy_path", str(default_policy))
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("goal_x", 1.0)
        self.declare_parameter("goal_y", 1.0)
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("max_linear", 0.22)
        self.declare_parameter("max_angular", 2.0)
        self.policy = NeatPolicy(self.get_parameter("policy_path").value)
        self.scan = None; self.odom = None; self.prev_action = [0.0, 0.0]; self.steps = 0
        self.pub = self.create_publisher(Twist, self.get_parameter("cmd_vel_topic").value, 10)
        self.create_subscription(LaserScan, self.get_parameter("scan_topic").value, self._scan, 10)
        self.create_subscription(Odometry, self.get_parameter("odom_topic").value, self._odom, 10)
        hz = max(1.0, float(self.get_parameter("rate_hz").value))
        self.timer = self.create_timer(1.0 / hz, self._tick)
        self.get_logger().info(f"Loaded TurtleBot policy ({len(self.policy.input_ids)} inputs). Waiting for scan and odometry.")

    def _scan(self, msg): self.scan = msg
    def _odom(self, msg): self.odom = msg

    def _tick(self):
        if self.scan is None or self.odom is None: return
        p = self.odom.pose.pose.position; q = self.odom.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w*q.z + q.x*q.y), 1.0 - 2.0 * (q.y*q.y + q.z*q.z))
        gx = float(self.get_parameter("goal_x").value) - p.x
        gy = float(self.get_parameter("goal_y").value) - p.y
        lx = math.cos(yaw)*gx + math.sin(yaw)*gy
        ly = -math.sin(yaw)*gx + math.cos(yaw)*gy
        distance = math.hypot(lx, ly); bearing = math.atan2(ly, lx)
        scan = self.scan; ranges=[]
        for i in range(24):
            angle = i * 2.0 * math.pi / 24.0
            index = round((angle - scan.angle_min) / scan.angle_increment) if scan.angle_increment else 0
            value = scan.ranges[index] if 0 <= index < len(scan.ranges) else scan.range_max
            if not math.isfinite(value): value = scan.range_max
            ranges.append(max(0.0, min(3.5, value)) / 3.5)
        v = self.odom.twist.twist.linear.x / 0.22
        w = self.odom.twist.twist.angular.z / 2.0
        obs = ranges + [min(distance,4.0)/4.0, math.sin(bearing), math.cos(bearing), v, w,
                        self.prev_action[0], self.prev_action[1], min(1.0,self.steps/300.0)]
        action = self.policy.act(obs); self.prev_action = action; self.steps += 1
        cmd = Twist(); cmd.linear.x = action[0] * min(0.22,float(self.get_parameter("max_linear").value))
        cmd.angular.z = action[1] * min(2.0,float(self.get_parameter("max_angular").value))
        self.pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args); node=TurtleBotPolicyNode()
    try: rclpy.spin(node)
    finally:
        node.pub.publish(Twist()); node.destroy_node(); rclpy.shutdown()
if __name__ == "__main__": main()
