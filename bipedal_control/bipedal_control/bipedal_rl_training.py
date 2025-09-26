#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
# Removed: from std_srvs.srv import Empty 
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import math
import time
from threading import Lock

# Global flag to control episode termination based on user input
# NOTE: This is a simple, non-blocking way to track user intent in the main loop
GLOBAL_QUIT_FLAG = False

class BipedalRLEnv(gym.Env, Node):
    """
    Minimal RL Environment for Bipedal Walking
    Uses only IMU orientation and angular velocity as state
    """
    
    def __init__(self):
        # Initialize ROS2 Node
        Node.__init__(self, 'bipedal_rl_node')
        gym.Env.__init__(self) # Initialize Gym Environment

        # ROS2 Setup
        # Renamed callback to avoid potential Python issues
        self.imu_sub = self.create_subscription(Imu, '/bipedal/imu_remapped', self._imu_callback, 10) 
        self.joint_pub = self.create_publisher(JointTrajectory, '/bipedal_controller/joint_trajectory', 10)
        
        # Removed: Service client initialization
        
        # State variables
        self.current_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        self.current_angular_vel = np.array([0.0, 0.0, 0.0])
        self.data_lock = Lock()
        self.data_received = False
        
        # RL Environment Setup
        self.observation_space = spaces.Box(low=np.array([-3.14, -3.14, -10.0, -10.0]), high=np.array([3.14, 3.14, 10.0, 10.0]), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1.5, -0.1, -1.5, -0.1]), high=np.array([1.5, 2.0, 1.5, 2.0]), dtype=np.float32)
        
        # Episode management
        self.episode_steps = 0
        self.max_episode_steps = 200
        self.episode_start_time = None
        
        self.get_logger().info('BipedalRLEnv initialized!')
        
    def _imu_callback(self, msg): # Renamed from imu_callback
        with self.data_lock:
            self.current_orientation = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
            self.current_angular_vel = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
            self.data_received = True
    
    # ... (quaternion_to_euler, get_observation, send_joint_command, calculate_reward, is_done methods remain the same) ...

    def quaternion_to_euler(self, x, y, z, w):
        """Convert quaternion to roll, pitch, yaw"""
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        pitch = math.asin(np.clip(sinp, -1.0, 1.0))
        return roll, pitch
    
    def get_observation(self):
        """Get current state: [pitch, roll, pitch_rate, roll_rate]"""
        with self.data_lock:
            if not self.data_received:
                return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            roll, pitch = self.quaternion_to_euler(*self.current_orientation)
            roll_rate = self.current_angular_vel[0]
            pitch_rate = self.current_angular_vel[1]
            return np.array([pitch, roll, pitch_rate, roll_rate], dtype=np.float32)
    
    def send_joint_command(self, actions):
        """Send joint commands to robot"""
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = ['left_hip_joint', 'left_knee_joint', 'right_hip_joint', 'right_knee_joint']
        point = JointTrajectoryPoint()
        point.positions = [float(actions[0]), float(actions[1]), float(actions[2]), float(actions[3])]
        point.time_from_start = Duration(sec=0, nanosec=100000000)
        trajectory_msg.points.append(point)
        self.joint_pub.publish(trajectory_msg)
    
    def calculate_reward(self, obs):
        """Calculate reward based on staying upright"""
        pitch, roll, pitch_rate, roll_rate = obs
        upright_reward = 1.0 - (abs(pitch) + abs(roll)) / 3.14
        stability_penalty = -(abs(pitch_rate) + abs(roll_rate)) * 0.1
        survival_reward = 0.1
        total_reward = upright_reward + stability_penalty + survival_reward
        return max(total_reward, -1.0)
    
    def is_done(self, obs):
        """Check if episode should end"""
        pitch, roll, pitch_rate, roll_rate = obs
        
        # Episode ends if robot falls over too much
        if abs(pitch) > 1.57 or abs(roll) > 1.57:
            self.get_logger().warn(f"Fallen! Pitch: {pitch:.2f}, Roll: {roll:.2f}")
            return True
            
        # Episode ends after max steps
        if self.episode_steps >= self.max_episode_steps:
            return True
            
        # Check global flag for user quit command ('q')
        global GLOBAL_QUIT_FLAG
        if GLOBAL_QUIT_FLAG:
            return True

        return False
    
    def reset(self, seed=None, options=None):
        """Reset environment for new episode (NO SIM RESET)"""
        super().reset(seed=seed)
        
        self.episode_steps = 0
        self.episode_start_time = time.time()
        
        # Removed: ROS service call for simulation reset
        self.get_logger().info('--- Waiting for Manual Simulation Reset and Joint Homing ---')
        
        # Reset robot to standing position
        reset_actions = np.array([0.0, 0.0, 0.0, 0.0])
        self.send_joint_command(reset_actions)
        
        # Note: The user MUST manually reset the physical simulation (Gazebo) now.
        time.sleep(1.0) # Wait for joints to move

        # Spin ROS to get fresh data
        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.1)
            
        obs = self.get_observation()
        info = {}
        
        return obs, info
    
    def step(self, action):
        """Take one step in environment"""
        # Send action to robot
        self.send_joint_command(action)
        
        # Wait for action to execute
        time.sleep(0.2)
        
        # Get fresh sensor data
        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.02)
        
        # Get new observation
        obs = self.get_observation()
        
        # Calculate reward
        reward = self.calculate_reward(obs)
        
        # Check if episode is done
        done = self.is_done(obs)
        truncated = False
        
        # Episode info
        info = {'episode_step': self.episode_steps, 'pitch': obs[0], 'roll': obs[1]}
        
        self.episode_steps += 1
        
        if self.episode_steps % 50 == 0:
            self.get_logger().info(
                f'Step {self.episode_steps}: Pitch={obs[0]:.3f}, Roll={obs[1]:.3f}, Reward={reward:.3f}'
            )
        
        return obs, reward, done, truncated, info


def train_bipedal_robot():
    """Main training function with 's' to start and 'q' to end episode."""
    global GLOBAL_QUIT_FLAG
    rclpy.init()
    
    try:
        env = BipedalRLEnv()
        
        # --- INITIAL PAUSE FOR USER TO START SIM/SETUP ---
        print("\n=======================================================")
        print("  Bipedal RL Training Program - Manual Control Mode")
        print("  Press 's' then ENTER to START the training loop.")
        print("  Press 'q' then ENTER to QUIT the entire program.")
        print("=======================================================")
        
        while True:
            cmd = input("Command: ").lower()
            if cmd == 's':
                print("STARTING TRAINING LOOP...")
                break
            elif cmd == 'q':
                print("Quitting program as requested.")
                return 
            else:
                print("Invalid command. Press 's' to start or 'q' to quit.")

        # --- WAIT FOR IMU DATA (Existing Logic) ---
        print("Waiting for IMU data...")
        while not env.data_received and rclpy.ok():
            rclpy.spin_once(env, timeout_sec=0.1)
            time.sleep(0.1)
        
        print("IMU data received! Starting training...")
        
        model = PPO(
            "MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=128, 
            batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, 
            clip_range=0.2, tensorboard_log="./bipedal_logs/"
        )
        
        total_timesteps = 10000
        print(f"Training for {total_timesteps} timesteps...")
        
        obs, info = env.reset()
        episode_rewards = []
        episode_reward = 0
        episode_count = 0
        
        for step in range(total_timesteps):
            # --- Check for 'q' to end the CURRENT episode ---
            # NOTE: We use input() to check for 'q' *before* taking the step.
            # This requires you to press 'q' and ENTER immediately before the next step.
            try:
                cmd = input(f"Step {step}/{total_timesteps}. Command ('q' to end episode, ENTER to continue): ").lower()
                if cmd == 'q':
                    print("\n'q' received. Terminating current episode...")
                    GLOBAL_QUIT_FLAG = True # Set flag to end the current episode
            except EOFError:
                pass
            except Exception:
                pass
            
            # Check if the global flag was set (by user input) or if robot fell (is_done)
            if GLOBAL_QUIT_FLAG or not rclpy.ok():
                 break # Exit the main loop completely if 'q' was pressed

            # Get action from policy
            action, _ = model.predict(obs.astype(np.float32), deterministic=False)
            
            # Take step
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            if done or truncated:
                episode_rewards.append(episode_reward)
                episode_count += 1
                
                avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
                print(f"\n--- Episode {episode_count} Finished (Reward={episode_reward:.2f}, Avg={avg_reward:.2f}) ---")
                
                # Reset the quit flag for the next episode
                GLOBAL_QUIT_FLAG = False 
                
                # NOTE: You MUST manually reset your simulation (e.g., Gazebo) now before training continues.
                
                obs, info = env.reset()
                episode_reward = 0
        
        # Save the model
        model.save("bipedal_walking_model")
        print("Model saved as 'bipedal_walking_model.zip'")
        
    except KeyboardInterrupt:
        print("Training interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        if 'env' in locals() and isinstance(env, Node):
             env.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    train_bipedal_robot()