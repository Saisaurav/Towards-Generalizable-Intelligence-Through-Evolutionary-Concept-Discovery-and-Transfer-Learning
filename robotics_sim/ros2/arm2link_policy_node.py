"""ROS 2 bridge for the 2-link arm feat=3 policy.

Subscribes to sensor_msgs/JointState. Publishes std_msgs/Float64MultiArray
joint velocities (rad/s) to a velocity-controller command topic.
"""
from __future__ import annotations
import math
from pathlib import Path
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from neat_policy_runtime import NeatPolicy

class Arm2LinkPolicyNode(Node):
    def __init__(self):
        super().__init__("frameforge_arm2link_policy")
        default_policy = Path(__file__).resolve().parents[1] / "policies" / "arm2link.json"
        self.declare_parameter("policy_path", str(default_policy))
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("command_topic", "/joint_group_velocity_controller/commands")
        self.declare_parameter("joint_1_name", "joint1")
        self.declare_parameter("joint_2_name", "joint2")
        self.declare_parameter("target_x", 0.38)
        self.declare_parameter("target_y", 0.12)
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("max_joint_velocity", 2.0)
        self.declare_parameter("filter_gain", 0.25)
        self.policy=NeatPolicy(self.get_parameter("policy_path").value)
        self.msg=None; self.prev_action=[0.0,0.0]; self.qhat=None; self.qf=[0.0,0.0]; self.steps=0
        self.pub=self.create_publisher(Float64MultiArray,self.get_parameter("command_topic").value,10)
        self.create_subscription(JointState,self.get_parameter("joint_states_topic").value,self._joints,10)
        hz=max(1.0,float(self.get_parameter("rate_hz").value));self.timer=self.create_timer(1.0/hz,self._tick)
        self.get_logger().info(f"Loaded arm policy ({len(self.policy.input_ids)} inputs). Waiting for joint states.")

    def _joints(self,msg): self.msg=msg
    @staticmethod
    def _fk(q1,q2): return (0.30*math.cos(q1)+0.25*math.cos(q1+q2),0.30*math.sin(q1)+0.25*math.sin(q1+q2))

    def _tick(self):
        if self.msg is None:return
        names=list(self.msg.name)
        n1=self.get_parameter("joint_1_name").value;n2=self.get_parameter("joint_2_name").value
        if n1 not in names or n2 not in names:return
        i1=names.index(n1);i2=names.index(n2);reading=[float(self.msg.position[i1]),float(self.msg.position[i2])]
        velocity=[float(self.msg.velocity[i1]) if len(self.msg.velocity)>i1 else 0.0,
                  float(self.msg.velocity[i2]) if len(self.msg.velocity)>i2 else 0.0]
        if self.qhat is None:self.qhat=reading[:]
        else:
            # Match training: q_pred = qhat_prev + prev_action * 2.0 * 0.05;
            # qhat = q_pred + fk * (measured_joint_angle - q_pred).
            pred=[self.qhat[k]+self.prev_action[k]*2.0*0.05 for k in range(2)]
            gain=float(self.get_parameter("filter_gain").value)
            self.qhat=[pred[k]+gain*(reading[k]-pred[k]) for k in range(2)]
        q1,q2=self.qhat; ex0,ey0=self._fk(q1,q2)
        ex=float(self.get_parameter("target_x").value)-ex0;ey=float(self.get_parameter("target_y").value)-ey0
        obs=[math.sin(q1),math.cos(q1),math.sin(q2),math.cos(q2),velocity[0]/2.0,velocity[1]/2.0,
             ex/0.55,ey/0.55,self.prev_action[0],self.prev_action[1],
             max(-1.0,min(1.0,ex/0.08)),max(-1.0,min(1.0,ey/0.08))]
        s1,c1=math.sin(q1),math.cos(q1);s12,c12=math.sin(q1+q2),math.cos(q1+q2)
        j11=-.30*s1-.25*s12;j12=-.25*s12;j21=.30*c1+.25*c12;j22=.25*c12
        obs.extend([max(-1.0,min(1.0,(j11*ex+j21*ey)/.05)),max(-1.0,min(1.0,(j12*ex+j22*ey)/.05)),min(1.0,self.steps/100.0)])
        action=self.policy.act(obs);self.prev_action=action;self.steps+=1
        cmd=Float64MultiArray();limit=min(2.0,float(self.get_parameter("max_joint_velocity").value))
        cmd.data=[action[0]*limit,action[1]*limit];self.pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args);node=Arm2LinkPolicyNode()
    try:rclpy.spin(node)
    finally:
        node.pub.publish(Float64MultiArray(data=[0.0,0.0]));node.destroy_node();rclpy.shutdown()
if __name__=="__main__":main()
