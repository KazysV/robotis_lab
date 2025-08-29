#!/usr/bin/env python3

# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Taehyeong Kim

import argparse

import numpy as np
import rclpy
from rclpy.node import Node

from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

from reach_env_cfg import ReachEnvConfig

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from sim2real.utils.policy_executor import PolicyExecutor


class OMYReachPolicy(Node, PolicyExecutor):
    """ROS2 node for executing a reach policy on the OMY robot."""

    def __init__(self, model_dir: str):
        super().__init__('omy_reach_policy_node')

        self.cfg = ReachEnvConfig(model_dir=model_dir)
        self.load_policy_model(self.cfg.policy_model_path)
        self.load_policy_yaml(self.cfg.policy_env_path)

        self.br = TransformBroadcaster(self)
        self.iteration = 0
        self.has_joint_data = False

        self.action_scale = self.get_action_scale()
        self.joint_names = self.get_observation_joint_names()
        self.default_pos = self.get_default_joint_positions(self.joint_names)

        self.target_command = np.zeros(7)  # [x, y, z, qw, qx, qy, qz]
        self.num_joints = len(self.joint_names)
        self.previous_action = np.zeros(self.num_joints)
        self.current_joint_positions = np.zeros(self.num_joints)
        self.current_joint_velocities = np.zeros(self.num_joints)

        self.joint_state_subscriber = self.create_subscription(
            JointState,
            self.cfg.joint_state_topic,
            self.joint_state_callback,
            10
        )
        self.joint_trajectory_publisher = self.create_publisher(
            JointTrajectory,
            self.cfg.joint_trajectory_topic,
            10
        )
        self.joint_command_timer = self.create_timer(self.cfg.step_size, self.timer_callback)

        self.get_logger().info("OMYReachPolicy node initialized.")

    def joint_state_callback(self, msg: JointState):
        """Update current joint state using only joints that exist in the message."""
        name_to_index = {name: i for i, name in enumerate(msg.name)}
        for i, name in enumerate(self.joint_names):
            if name in name_to_index:
                idx = name_to_index[name]
                self.current_joint_positions[i] = msg.position[idx]
                if idx < len(msg.velocity):
                    self.current_joint_velocities[i] = msg.velocity[idx]
                else:
                    self.current_joint_velocities[i] = 0.0
            else:
                self.get_logger().warn(f"Joint '{name}' not found in JointState message. Using previous value.")
        self.has_joint_data = True

    def timer_callback(self):
        """Main control loop: samples target, computes action, and publishes joint commands."""
        if not self.has_joint_data:
            return

        command_interval = int(self.cfg.send_command_interval / self.cfg.step_size)  # interval in steps
        phase = self.iteration % (2 * command_interval)

        if phase == 0:
            self.target_command = self.cfg.sample_random_pose()
            self.broadcast_target_pose_tf()
            self.get_logger().info(f"New target command: {np.round(self.target_command, 4)}")

        if phase < command_interval:
            joint_trajectory_msg = self.create_trajectory_command(self.default_pos)
            self.joint_trajectory_publisher.publish(joint_trajectory_msg)
        else:
            joint_positions = self.run_policy_step(self.target_command)
            if len(joint_positions) != self.num_joints:
                raise ValueError(f"Expected {self.num_joints} joint positions, got {len(joint_positions)}")
            joint_trajectory_msg = self.create_trajectory_command(joint_positions)
            self.joint_trajectory_publisher.publish(joint_trajectory_msg)

        self.iteration += 1

    def create_trajectory_command(self, joint_positions: np.ndarray) -> JointTrajectory:
        """Creates a JointTrajectory message from joint positions."""
        point = JointTrajectoryPoint()
        point.positions = joint_positions.tolist()
        point.time_from_start = Duration(
            sec=0,
            nanosec=int(self.cfg.trajectory_time_from_start * 1e9)
        )

        joint_trajectory = JointTrajectory()
        joint_trajectory.joint_names = self.joint_names
        joint_trajectory.points.append(point)
        return joint_trajectory

    def broadcast_target_pose_tf(self):
        """Publishes a TF transform for the target pose."""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "world"
        t.child_frame_id = "target_pose"

        t.transform.translation.x = self.target_command[0]
        t.transform.translation.y = self.target_command[1]
        t.transform.translation.z = self.target_command[2]
        t.transform.rotation.x = self.target_command[4]
        t.transform.rotation.y = self.target_command[5]
        t.transform.rotation.z = self.target_command[6]
        t.transform.rotation.w = self.target_command[3]

        self.br.sendTransform(t)

    def update_observation(self, command: np.ndarray) -> np.ndarray:
        """Builds the observation vector for the policy."""
        obs = np.concatenate([
            self.current_joint_positions - self.default_pos,
            self.current_joint_velocities,
            command,
            self.previous_action,
        ]).astype(np.float32)

        return obs

    def run_policy_step(self, command: np.ndarray) -> np.ndarray:
        """Runs a single step of the policy and returns the joint positions to command."""
        observation = self.update_observation(command)
        self.action = self.update_action(observation)
        self.previous_action = self.action.copy()
        joint_positions = self.default_pos + (self.action * self.action_scale)

        return joint_positions


def main(args=None):
    """Entry point to initialize ROS2 and run the reach policy node."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", type=str, required=True,
        help="Relative path to the trained policy directory under logs/rsl_rl/reach_omy/"
    )

    parsed_args, remaining_args = parser.parse_known_args(args)

    rclpy.init(args=remaining_args)
    node = OMYReachPolicy(model_dir=parsed_args.model_dir)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
