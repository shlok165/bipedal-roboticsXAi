#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import math
import time
import threading
from threading import Lock
import select
import sys
import termios
import tty
import subprocess
import os

class BipedalRLEnv(gym.Env, Node):
    """
    Enhanced Bipedal RL Environment with fall detection and auto-restart
    """
    
    def __init__(self):
        # Initialize ROS2 Node
        Node.__init__(self, 'bipedal_rl_node')
        
        # ROS2 Setup
        self.imu_sub = self.create_subscription(Imu, '/bipedal/imu_remapped', self.imu_callback, 10)
        self.joint_pub = self.create_publisher(JointTrajectory, '/bipedal_controller/joint_trajectory', 10)
        
        # State variables
        self.current_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        self.current_angular_vel = np.array([0.0, 0.0, 0.0])
        self.current_linear_accel = np.array([0.0, 0.0, 9.81])  # Initial gravity
        self.data_lock = Lock()
        self.data_received = False
        
        # Fall detection parameters
        self.robot_standing_height = 0.8  # Assume ~80cm when standing normally
        self.current_estimated_height = self.robot_standing_height
        self.fall_threshold = 0.5  # 50cm - robot definitely fallen
        self.warning_threshold = 0.65  # 65cm - robot falling significantly
        self.velocity_z = 0.0  # Vertical velocity for height estimation
        self.last_time = time.time()
        
        # Fall detection state
        self.is_fallen = False
        self.fall_warning = False
        
        # RL Environment Setup - enhanced observation space
        self.observation_space = spaces.Box(
            low=np.array([-3.14, -3.14, -10.0, -10.0, 0.0, -1.0]), 
            high=np.array([3.14, 3.14, 10.0, 10.0, 2.0, 1.0]), 
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=np.array([-1.5, -0.1, -1.5, -0.1]), 
            high=np.array([1.5, 2.0, 1.5, 2.0]), 
            dtype=np.float32
        )
        
        # Episode management
        self.episode_steps = 0
        self.max_episode_steps = 500
        
        # Keyboard input handling
        self.key_pressed = False
        self.key_thread = threading.Thread(target=self._keyboard_listener)
        self.key_thread.daemon = True
        self.key_thread.start()
        
        # Create restart script
        self.create_restart_script()
        
        self.get_logger().info('Enhanced BipedalRLEnv initialized with fall detection!')
    
    def create_restart_script(self):
        """Create the restart script for ROS system"""
        script_content = '''#!/bin/bash

gz service -s /world/bipedal_world/set_pose --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --req '
name: "bipedal"
position: {x: 0.0, y: 0.0, z: 0.8}
orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
'
sleep 5
'''
        
        script_path = os.path.expanduser('~/restart_robot.sh')
        try:
            with open(script_path, 'w') as f:
                f.write(script_content)
            os.chmod(script_path, 0o755)  # Make executable
            self.restart_script_path = script_path
            self.get_logger().info(f'Restart script created at: {script_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to create restart script: {e}')
            self.restart_script_path = None
    
    def _keyboard_listener(self):
        """Listen for ENTER key press"""
        while rclpy.ok():
            try:
                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.readline().strip()
                    if key == '':
                        self.key_pressed = True
                        self.get_logger().info('ENTER key detected!')
                time.sleep(0.1)
            except:
                time.sleep(0.1)
    
    def imu_callback(self, msg):
        current_time = time.time()
        
        with self.data_lock:
            self.current_orientation = np.array([
                msg.orientation.x, msg.orientation.y, 
                msg.orientation.z, msg.orientation.w
            ])
            self.current_angular_vel = np.array([
                msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z
            ])
            self.current_linear_accel = np.array([
                msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
            ])
            
            # Update height estimation
            if self.data_received:  # Not first callback
                dt = current_time - self.last_time
                if dt > 0 and dt < 0.5:  # Reasonable time step
                    self.update_height_estimation(dt)
            
            self.last_time = current_time
            self.data_received = True
    
    def update_height_estimation(self, dt):
        """Estimate robot height using IMU data"""
        try:
            # Get world-frame acceleration (remove gravity)
            roll, pitch = self.quaternion_to_euler(*self.current_orientation)
            
            # Transform acceleration to world frame (simplified)
            # Remove gravity component assuming robot is roughly upright
            accel_world_z = self.current_linear_accel[2] * math.cos(pitch) * math.cos(roll) - 9.81
            
            # Update velocity and height using integration
            self.velocity_z += accel_world_z * dt
            height_change = self.velocity_z * dt + 0.5 * accel_world_z * dt * dt
            self.current_estimated_height += height_change
            
            # Apply some damping to prevent drift
            self.velocity_z *= 0.98
            
            # Clamp height to reasonable bounds
            self.current_estimated_height = max(0.1, min(1.2, self.current_estimated_height))
            
            # Check for fall conditions
            if self.current_estimated_height <= self.fall_threshold:
                if not self.is_fallen:
                    self.is_fallen = True
                    self.get_logger().warn(f'ROBOT FALLEN! Height: {self.current_estimated_height:.3f}m')
            elif self.current_estimated_height <= self.warning_threshold:
                if not self.fall_warning:
                    self.fall_warning = True
                    self.get_logger().warn(f'Fall warning! Height: {self.current_estimated_height:.3f}m')
            else:
                # Reset warnings if robot recovers
                if self.fall_warning and self.current_estimated_height > self.warning_threshold + 0.05:
                    self.fall_warning = False
                    self.get_logger().info('Robot recovered from fall warning')
                    
        except Exception as e:
            self.get_logger().error(f'Height estimation error: {e}')
    
    def quaternion_to_euler(self, x, y, z, w):
        """Convert quaternion to roll, pitch"""
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        return roll, pitch
    
    def get_observation(self):
        """Get current state including height information"""
        with self.data_lock:
            if not self.data_received:
                return np.array([0.0, 0.0, 0.0, 0.0, 0.8, 0.0], dtype=np.float32)
            
            roll, pitch = self.quaternion_to_euler(*self.current_orientation)
            roll_rate = self.current_angular_vel[0]
            pitch_rate = self.current_angular_vel[1]
            
            # Normalize height (0.0 = fallen, 1.0 = standing)
            height_normalized = (self.current_estimated_height - 0.2) / 0.6
            height_normalized = max(0.0, min(1.0, height_normalized))
            
            return np.array([
                pitch, roll, pitch_rate, roll_rate, 
                height_normalized, self.velocity_z
            ], dtype=np.float32)
    
    def send_joint_command(self, actions):
        """Send joint commands to robot"""
        try:
            trajectory_msg = JointTrajectory()
            trajectory_msg.joint_names = [
                'left_hip_joint', 'left_knee_joint',
                'right_hip_joint', 'right_knee_joint'
            ]
            
            point = JointTrajectoryPoint()
            point.positions = [float(actions[0]), float(actions[1]), 
                              float(actions[2]), float(actions[3])]
            point.time_from_start = Duration(sec=0, nanosec=100000000)
            
            trajectory_msg.points.append(point)
            self.joint_pub.publish(trajectory_msg)
        except Exception as e:
            self.get_logger().error(f'Failed to send joint command: {e}')
    
    def calculate_reward(self, obs):
        """Enhanced reward calculation including height"""
        pitch, roll, pitch_rate, roll_rate, height_norm, vel_z = obs
        
        # Base upright reward
        upright_reward = 1.0 - (abs(pitch) + abs(roll)) / 3.14
        
        # Stability penalty
        stability_penalty = -(abs(pitch_rate) + abs(roll_rate)) * 0.1
        
        # Height reward - heavily reward staying upright
        height_reward = height_norm * 2.0
        
        # Penalize falling velocity
        fall_velocity_penalty = -abs(vel_z) * 0.5 if vel_z < -0.1 else 0.0
        
        # Survival reward
        survival_reward = 0.1
        
        total_reward = upright_reward + stability_penalty + height_reward + fall_velocity_penalty + survival_reward
        
        # Heavy penalty for falling
        if self.is_fallen:
            total_reward -= 10.0
            
        return max(total_reward, -10.0)
    
    def restart_ros_system(self):
        """Restart the ROS system after a fall"""
        if self.restart_script_path and os.path.exists(self.restart_script_path):
            try:
                self.get_logger().info('Executing restart script...')
                result = subprocess.run([self.restart_script_path], 
                                      capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    self.get_logger().info('ROS system restart completed successfully')
                else:
                    self.get_logger().error(f'Restart script failed: {result.stderr}')
            except subprocess.TimeoutExpired:
                self.get_logger().error('Restart script timed out')
            except Exception as e:
                self.get_logger().error(f'Failed to execute restart script: {e}')
        else:
            self.get_logger().error('Restart script not found')
    
    def is_done(self, obs):
        """Enhanced episode termination logic"""
        pitch, roll, pitch_rate, roll_rate, height_norm, vel_z = obs
        
        # Manual reset
        if self.key_pressed:
            self.get_logger().info('Manual reset triggered by ENTER key!')
            self.key_pressed = False
            return True
        
        # Fall detection
        if self.is_fallen:
            self.get_logger().warn(f'Episode ended due to fall! Height: {self.current_estimated_height:.3f}m')
            return True
        
        # Extreme angle termination (backup)
        if abs(pitch) > 1.2 or abs(roll) > 1.2:
            self.get_logger().warn('Episode ended due to extreme angle!')
            return True
        
        # Max steps
        if self.episode_steps >= self.max_episode_steps:
            self.get_logger().info('Episode ended - max steps reached')
            return True
        
        return False
    
    def reset(self, seed=None, options=None):
        """Reset environment for new episode"""
        super().reset(seed=seed)
        
        # If robot fell, restart ROS system
        if self.is_fallen:
            self.get_logger().info('Restarting ROS system due to fall...')
            self.restart_ros_system()
            time.sleep(10)  # Wait for system to fully restart
        
        # Reset episode state
        self.episode_steps = 0
        self.is_fallen = False
        self.fall_warning = False
        self.current_estimated_height = self.robot_standing_height
        self.velocity_z = 0.0
        
        # Reset robot joints
        reset_actions = np.array([0.0, 0.0, 0.0, 0.0])
        self.send_joint_command(reset_actions)
        
        # Wait for reset and data
        time.sleep(2.0)
        
        # Get fresh data
        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.1)
        
        obs = self.get_observation()
        info = {
            'height': self.current_estimated_height,
            'fall_warning': self.fall_warning
        }
        
        self.get_logger().info(f'Episode reset complete. Height: {self.current_estimated_height:.3f}m')
        return obs, info
    
    def step(self, action):
        """Take one step in environment"""
        self.send_joint_command(action)
        time.sleep(0.1)
        
        # Get sensor updates
        for _ in range(3):
            rclpy.spin_once(self, timeout_sec=0.05)
        
        obs = self.get_observation()
        reward = self.calculate_reward(obs)
        done = self.is_done(obs)
        truncated = False
        
        info = {
            'episode_step': self.episode_steps,
            'pitch': obs[0],
            'roll': obs[1],
            'height': self.current_estimated_height,
            'height_normalized': obs[4],
            'fall_warning': self.fall_warning,
            'is_fallen': self.is_fallen
        }
        
        self.episode_steps += 1
        
        # Enhanced logging
        if self.episode_steps % 50 == 0 or done or self.fall_warning:
            self.get_logger().info(
                f'Step {self.episode_steps}: Pitch={obs[0]:.3f}, Roll={obs[1]:.3f}, '
                f'Height={self.current_estimated_height:.3f}m, Reward={reward:.3f}'
                f'{" [FALL WARNING]" if self.fall_warning else ""}'
                f'{" [FALLEN]" if self.is_fallen else ""}'
            )
        
        return obs, reward, done, truncated, info

def train_bipedal_robot():
    """Main training function"""
    old_settings = None
    try:
        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
    except:
        print("Warning: Non-blocking input not available")
    
    try:
        rclpy.init()
        env = BipedalRLEnv()
        
        print("Enhanced Bipedal RL Training Started!")
        print("Features:")
        print("- Fall detection based on height estimation")
        print("- Automatic ROS system restart after falls")
        print("- Press ENTER to manually reset episodes")
        print("- Ctrl+C to stop training")
        print(f"- Fall threshold: {env.fall_threshold}m")
        print(f"- Warning threshold: {env.warning_threshold}m")
        
        # Wait for data
        print("\nWaiting for IMU data...")
        start_time = time.time()
        while not env.data_received and (time.time() - start_time < 30.0):
            rclpy.spin_once(env, timeout_sec=0.1)
            time.sleep(0.1)
        
        if not env.data_received:
            print("ERROR: No IMU data received! Check your robot connection.")
            return
        
        print(f"IMU data received! Initial height: {env.current_estimated_height:.3f}m")
        
        # Create model with enhanced observation space
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./bipedal_logs/")
        
        # Train
        print("\nStarting training...")
        model.learn(total_timesteps=20000)
        model.save("bipedal_model_enhanced")
        print("Training completed!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if old_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except:
                pass
        rclpy.shutdown()

if __name__ == '__main__':
    train_bipedal_robot()