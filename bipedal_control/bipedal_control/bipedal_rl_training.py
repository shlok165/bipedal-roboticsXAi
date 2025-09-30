#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
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
    Improved Bipedal Walking RL Environment
    - Better observation space with joint states
    - Reward for forward movement
    - Smooth action control
    - Proper fall detection
    """
    
    def __init__(self):
        # Initialize ROS2 Node
        Node.__init__(self, 'bipedal_rl_node')
        
        # ROS2 Setup
        self.imu_sub = self.create_subscription(Imu, '/bipedal/imu_remapped', self.imu_callback, 10)
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.joint_pub = self.create_publisher(JointTrajectory, '/bipedal_controller/joint_trajectory', 10)
        
        # State variables
        self.current_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        self.current_angular_vel = np.array([0.0, 0.0, 0.0])
        self.current_linear_accel = np.array([0.0, 0.0, 9.81])
        
        # Joint states (6 joints)
        self.joint_names = ['left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
                           'right_hip_joint', 'right_knee_joint', 'right_ankle_joint']
        self.current_joint_positions = np.zeros(6)
        self.current_joint_velocities = np.zeros(6)
        self.last_joint_positions = np.zeros(6)
        
        # Data synchronization
        self.data_lock = Lock()
        self.imu_received = False
        self.joint_received = False
        
        # Position tracking for forward movement reward
        self.initial_x_position = 0.0
        self.current_x_position = 0.0
        self.last_x_position = 0.0
        self.x_velocity = 0.0
        
        # Fall detection
        self.is_fallen = False
        self.fall_angle_threshold = 0.8  # ~45 degrees
        
        # Enhanced observation space:
        # [pitch, roll, pitch_rate, roll_rate, 
        #  6 joint positions, 6 joint velocities,
        #  x_velocity, phase]
        obs_dim = 4 + 6 + 6 + 2  # 18 total
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action space: delta changes to joint positions (smoother control)
        self.action_space = spaces.Box(
            low=-0.2, high=0.2, shape=(6,), dtype=np.float32
        )
        
        # Target joint positions (will be modified by actions)
        self.target_joint_positions = np.zeros(6)
        
        # Episode management
        self.episode_steps = 0
        self.max_episode_steps = 1000  # Longer episodes for learning walking
        self.episode_number = 0
        self.episode_start_time = time.time()
        
        # Walking phase (for cyclical reward)
        self.walking_phase = 0.0
        
        # Curriculum learning parameters
        self.curriculum_stage = 0
        self.success_count = 0
        
        # Keyboard input handling
        self.key_pressed = False
        self.key_thread = threading.Thread(target=self._keyboard_listener)
        self.key_thread.daemon = True
        self.key_thread.start()
        
        # Create restart script
        self.create_restart_script()
        
        self.get_logger().info('Improved BipedalRLEnv initialized!')
    
    def create_restart_script(self):
        """Create the restart script for ROS system"""
        script_content = '''#!/bin/bash

ros2 topic pub /bipedal_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory "
{       
  joint_names: ['left_hip_joint', 'left_knee_joint', 'left_ankle_joint', 'right_hip_joint', 'right_knee_joint', 'right_ankle_joint'],
  points: [
    {
      positions: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      time_from_start: {sec: 0, nanosec: 0}
    }
  ]
}" --once

sleep 2

gz service -s /world/bipedal_world/set_pose --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --req '
name: "bipedal"
position: {x: 0.0, y: 0.0, z: 0.8}
orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
'
sleep 2
'''
        
        script_path = os.path.expanduser('~/restart_robot.sh')
        try:
            with open(script_path, 'w') as f:
                f.write(script_content)
            os.chmod(script_path, 0o755)
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
            self.imu_received = True
    
    def joint_callback(self, msg):
        """Callback for joint states"""
        with self.data_lock:
            try:
                # Map joint names to our order
                joint_dict = {name: (pos, vel) for name, pos, vel in 
                            zip(msg.name, msg.position, msg.velocity)}
                
                for i, joint_name in enumerate(self.joint_names):
                    if joint_name in joint_dict:
                        self.current_joint_positions[i] = joint_dict[joint_name][0]
                        self.current_joint_velocities[i] = joint_dict[joint_name][1]
                
                self.joint_received = True
            except Exception as e:
                self.get_logger().error(f'Joint callback error: {e}')
    
    def quaternion_to_euler(self, x, y, z, w):
        """Convert quaternion to roll, pitch, yaw"""
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    def estimate_x_position(self):
        """Estimate x position from IMU acceleration"""
        # Simple integration of x-acceleration
        # This is rough but better than nothing without odometry
        dt = 0.1
        ax = self.current_linear_accel[0]
        
        # Simple velocity estimation
        self.x_velocity = self.x_velocity * 0.95 + ax * dt  # With damping
        self.current_x_position += self.x_velocity * dt
    
    def get_observation(self):
        """Get current state observation"""
        with self.data_lock:
            if not (self.imu_received and self.joint_received):
                return np.zeros(18, dtype=np.float32)
            
            roll, pitch, yaw = self.quaternion_to_euler(*self.current_orientation)
            roll_rate = self.current_angular_vel[0]
            pitch_rate = self.current_angular_vel[1]
            
            # Update position estimate
            self.estimate_x_position()
            
            # Build observation vector
            obs = np.concatenate([
                [pitch, roll, pitch_rate, roll_rate],  # IMU data
                self.current_joint_positions,  # Joint positions
                self.current_joint_velocities,  # Joint velocities
                [self.x_velocity, self.walking_phase]  # Movement and phase
            ]).astype(np.float32)
            
            return obs
    
    def send_joint_command(self, positions):
        """Send joint commands to robot"""
        try:
            trajectory_msg = JointTrajectory()
            trajectory_msg.joint_names = self.joint_names
            
            point = JointTrajectoryPoint()
            point.positions = [float(p) for p in positions]
            point.time_from_start = Duration(sec=0, nanosec=100000000)  # 0.1s
            
            trajectory_msg.points.append(point)
            self.joint_pub.publish(trajectory_msg)
        except Exception as e:
            self.get_logger().error(f'Failed to send joint command: {e}')
    
    def calculate_reward(self, obs, action):
        """Calculate reward based on multiple factors"""
        pitch, roll, pitch_rate, roll_rate = obs[0:4]
        joint_positions = obs[4:10]
        joint_velocities = obs[10:16]
        x_velocity, phase = obs[16:18]
        
        reward = 0.0
        
        # 1. Upright reward (most important)
        upright_reward = 1.0 - (abs(pitch) + abs(roll)) / math.pi
        reward += upright_reward * 2.0
        
        # 2. Forward movement reward (main goal)
        forward_reward = max(0, x_velocity)  # Reward positive velocity
        reward += forward_reward * 5.0
        
        # 3. Stability penalty (penalize high angular velocities)
        stability_penalty = -(abs(pitch_rate) + abs(roll_rate)) * 0.1
        reward += stability_penalty
        
        # 4. Energy efficiency (penalize large actions)
        energy_penalty = -np.sum(np.abs(action)) * 0.01
        reward += energy_penalty
        
        # 5. Joint velocity penalty (smoother is better)
        smoothness_penalty = -np.sum(np.abs(joint_velocities)) * 0.01
        reward += smoothness_penalty
        
        # 6. Symmetric gait reward (legs should move opposite)
        left_hip, left_knee = joint_positions[0], joint_positions[1]
        right_hip, right_knee = joint_positions[3], joint_positions[4]
        symmetry = -abs((left_hip + right_hip)) * 0.1  # Hips should be opposite
        reward += symmetry
        
        # 7. Knee bending reward (encourage dynamic walking)
        knee_bend_reward = (abs(left_knee) + abs(right_knee)) * 0.2
        reward += knee_bend_reward
        
        # 8. Survival bonus
        reward += 0.1
        
        # 9. Curriculum-based reward
        if self.curriculum_stage == 0:
            # Stage 0: Just stay upright
            reward += upright_reward * 3.0
        elif self.curriculum_stage == 1:
            # Stage 1: Stay upright and move forward
            reward += forward_reward * 3.0
        else:
            # Stage 2: Optimize gait
            reward += forward_reward * 5.0
        
        # Heavy penalty for falling
        if self.is_fallen:
            reward -= 20.0
        
        return reward
    
    def restart_ros_system(self):
        """Restart the ROS system after a fall"""
        if self.restart_script_path and os.path.exists(self.restart_script_path):
            try:
                self.get_logger().info('Executing restart script...')
                result = subprocess.run([self.restart_script_path], 
                                      capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    self.get_logger().info('ROS system restart completed')
                else:
                    self.get_logger().error(f'Restart failed: {result.stderr}')
            except Exception as e:
                self.get_logger().error(f'Failed to execute restart: {e}')
        else:
            self.get_logger().error('Restart script not found')
    
    def is_done(self, obs):
        """Check if episode should end"""
        pitch, roll = obs[0], obs[1]
        
        # Manual reset
        if self.key_pressed:
            self.get_logger().info('Manual reset triggered!')
            self.key_pressed = False
            return True
        
        # Fall detection (robot tilted too much)
        if abs(pitch) > self.fall_angle_threshold or abs(roll) > self.fall_angle_threshold:
            if not self.is_fallen:
                self.is_fallen = True
                self.get_logger().warn(f'Episode {self.episode_number} - Robot fell! Pitch={pitch:.2f}, Roll={roll:.2f}')
            return True
        
        # Max steps
        if self.episode_steps >= self.max_episode_steps:
            self.get_logger().info(f'Episode {self.episode_number} completed successfully!')
            self.success_count += 1
            return True
        
        return False
    
    def reset(self, seed=None, options=None):
        """Reset environment for new episode"""
        super().reset(seed=seed)
        
        # If robot fell, restart system
        if self.is_fallen:
            self.get_logger().info('Restarting ROS system...')
            self.restart_ros_system()
            time.sleep(3)
        
        # Update curriculum if needed
        if self.episode_number > 0 and self.episode_number % 20 == 0:
            if self.success_count >= 10 and self.curriculum_stage < 2:
                self.curriculum_stage += 1
                self.get_logger().info(f'Advanced to curriculum stage {self.curriculum_stage}')
                self.success_count = 0
        
        # Reset episode state
        self.episode_number += 1
        self.episode_steps = 0
        self.is_fallen = False
        self.walking_phase = 0.0
        self.episode_start_time = time.time()
        
        # Reset position tracking
        self.initial_x_position = 0.0
        self.current_x_position = 0.0
        self.last_x_position = 0.0
        self.x_velocity = 0.0
        
        # Reset to neutral standing pose
        self.target_joint_positions = np.zeros(6)
        self.send_joint_command(self.target_joint_positions)
        
        # Wait for system to stabilize
        time.sleep(2.0)
        
        # Get fresh sensor data
        for _ in range(30):
            rclpy.spin_once(self, timeout_sec=0.05)
        
        obs = self.get_observation()
        info = {
            'curriculum_stage': self.curriculum_stage,
            'success_count': self.success_count
        }
        
        self.get_logger().info(f'Episode {self.episode_number} started (Stage {self.curriculum_stage})')
        return obs, info
    
    def step(self, action):
        """Take one step in environment"""
        # Apply action as delta to target positions
        self.target_joint_positions += action
        
        # Clamp joint positions to safe ranges
        # Hip: -1.5 to 1.5, Knee: -0.1 to 2.0, Ankle: -0.5 to 0.5
        joint_limits = np.array([
            [-1.5, 1.5],  # left hip
            [-0.1, 2.0],  # left knee
            [-0.5, 0.5],  # left ankle
            [-1.5, 1.5],  # right hip
            [-0.1, 2.0],  # right knee
            [-0.5, 0.5]   # right ankle
        ])
        
        for i in range(6):
            self.target_joint_positions[i] = np.clip(
                self.target_joint_positions[i], 
                joint_limits[i][0], 
                joint_limits[i][1]
            )
        
        # Send command
        self.send_joint_command(self.target_joint_positions)
        time.sleep(0.1)
        
        # Update walking phase
        self.walking_phase = (self.walking_phase + 0.1) % (2 * math.pi)
        
        # Get sensor updates
        for _ in range(3):
            rclpy.spin_once(self, timeout_sec=0.03)
        
        obs = self.get_observation()
        reward = self.calculate_reward(obs, action)
        done = self.is_done(obs)
        truncated = False
        
        info = {
            'episode_step': self.episode_steps,
            'x_position': self.current_x_position,
            'x_velocity': self.x_velocity,
            'curriculum_stage': self.curriculum_stage
        }
        
        self.episode_steps += 1
        
        # Logging
        if self.episode_steps % 100 == 0 or done:
            elapsed = time.time() - self.episode_start_time
            self.get_logger().info(
                f'Ep {self.episode_number}, Step {self.episode_steps}: '
                f'X={self.current_x_position:.2f}m, V={self.x_velocity:.3f}m/s, '
                f'Reward={reward:.2f}, Time={elapsed:.1f}s'
            )
        
        return obs, reward, done, truncated, info


class ProgressCallback(BaseCallback):
    """Custom callback for logging training progress"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self):
        if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer) > len(self.episode_rewards):
            info = self.model.ep_info_buffer[-1]
            self.episode_rewards.append(info['r'])
            self.episode_lengths.append(info['l'])
            
            if len(self.episode_rewards) % 10 == 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                mean_length = np.mean(self.episode_lengths[-10:])
                print(f"\n[Callback] Last 10 episodes - Mean reward: {mean_reward:.2f}, Mean length: {mean_length:.1f}")
        
        return True


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
        
        print("\n" + "="*60)
        print("Improved Bipedal Walking RL Training")
        print("="*60)
        print("\nFeatures:")
        print("  • Joint state feedback for better control")
        print("  • Forward movement reward")
        print("  • Curriculum learning (3 stages)")
        print("  • Smooth action control (delta positions)")
        print("  • Energy efficiency optimization")
        print("  • Symmetric gait encouragement")
        print("\nControls:")
        print("  • Press ENTER to manually reset")
        print("  • Press Ctrl+C to stop training")
        print("="*60)
        
        # Wait for sensor data
        print("\nWaiting for sensor data...")
        start_time = time.time()
        while (not env.imu_received or not env.joint_received) and (time.time() - start_time < 30.0):
            rclpy.spin_once(env, timeout_sec=0.1)
            time.sleep(0.1)
        
        if not env.imu_received or not env.joint_received:
            print("ERROR: Sensor data not received!")
            print(f"  IMU: {env.imu_received}, Joints: {env.joint_received}")
            return
        
        print("✓ Sensor data received!")
        print(f"  IMU orientation: {env.current_orientation}")
        print(f"  Joint positions: {env.current_joint_positions}")
        
        # Create PPO model with optimized hyperparameters
        print("\nInitializing PPO model...")
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            tensorboard_log="./bipedal_walking_logs/"
        )
        
        # Train with callback
        print("\nStarting training...\n")
        callback = ProgressCallback()
        model.learn(total_timesteps=50000, callback=callback, progress_bar=True)
        
        # Save model
        model_path = "bipedal_walker_improved"
        model.save(model_path)
        print(f"\n✓ Training completed! Model saved to {model_path}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if old_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except:
                pass
        try:
            rclpy.shutdown()
        except:
            pass


if __name__ == '__main__':
    train_bipedal_robot()