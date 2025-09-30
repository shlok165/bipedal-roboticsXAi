#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

import numpy as np
from stable_baselines3 import PPO
import math
import time
from threading import Lock
import sys

class BipedalWalkerRunner(Node):
    """
    Run the trained bipedal walking model
    """
    
    def __init__(self, model_path):
        super().__init__('bipedal_walker_runner')
        
        # Load the trained model
        try:
            self.model = PPO.load(model_path)
            self.get_logger().info(f'✓ Model loaded from: {model_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            raise
        
        # ROS2 Setup
        self.imu_sub = self.create_subscription(Imu, '/bipedal/imu_remapped', self.imu_callback, 10)
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        self.joint_pub = self.create_publisher(JointTrajectory, '/bipedal_controller/joint_trajectory', 10)
        
        # State variables
        self.current_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        self.current_angular_vel = np.array([0.0, 0.0, 0.0])
        self.current_linear_accel = np.array([0.0, 0.0, 9.81])
        
        # Joint states
        self.joint_names = ['left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
                           'right_hip_joint', 'right_knee_joint', 'right_ankle_joint']
        self.current_joint_positions = np.zeros(6)
        self.current_joint_velocities = np.zeros(6)
        
        # Data synchronization
        self.data_lock = Lock()
        self.imu_received = False
        self.joint_received = False
        
        # Position tracking
        self.x_velocity = 0.0
        self.walking_phase = 0.0
        
        # Target joint positions
        self.target_joint_positions = np.zeros(6)
        
        # Statistics
        self.step_count = 0
        self.start_time = time.time()
        
        # Control loop timer (10 Hz = 0.1 seconds)
        self.control_timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('Walker runner initialized!')
        self.get_logger().info('Press Ctrl+C to stop')
    
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
        with self.data_lock:
            try:
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
        """Estimate x velocity from IMU"""
        dt = 0.1
        ax = self.current_linear_accel[0]
        self.x_velocity = self.x_velocity * 0.95 + ax * dt
    
    def get_observation(self):
        """Get current state observation"""
        with self.data_lock:
            if not (self.imu_received and self.joint_received):
                return np.zeros(18, dtype=np.float32)
            
            roll, pitch, yaw = self.quaternion_to_euler(*self.current_orientation)
            roll_rate = self.current_angular_vel[0]
            pitch_rate = self.current_angular_vel[1]
            
            self.estimate_x_position()
            
            obs = np.concatenate([
                [pitch, roll, pitch_rate, roll_rate],
                self.current_joint_positions,
                self.current_joint_velocities,
                [self.x_velocity, self.walking_phase]
            ]).astype(np.float32)
            
            return obs
    
    def send_joint_command(self, positions):
        """Send joint commands to robot"""
        try:
            trajectory_msg = JointTrajectory()
            trajectory_msg.joint_names = self.joint_names
            
            point = JointTrajectoryPoint()
            point.positions = [float(p) for p in positions]
            point.time_from_start = Duration(sec=0, nanosec=100000000)
            
            trajectory_msg.points.append(point)
            self.joint_pub.publish(trajectory_msg)
        except Exception as e:
            self.get_logger().error(f'Failed to send joint command: {e}')
    
    def control_loop(self):
        """Main control loop - called at 10 Hz"""
        if not (self.imu_received and self.joint_received):
            self.get_logger().warn('Waiting for sensor data...', throttle_duration_sec=2.0)
            return
        
        # Get current observation
        obs = self.get_observation()
        
        # Get action from trained model
        action, _states = self.model.predict(obs, deterministic=True)
        
        # Apply action (delta to target positions)
        self.target_joint_positions += action
        
        # Clamp to joint limits
        joint_limits = np.array([
            [-1.5, 1.5],   # left hip
            [-0.1, 2.0],   # left knee
            [-0.5, 0.5],   # left ankle
            [-1.5, 1.5],   # right hip
            [-0.1, 2.0],   # right knee
            [-0.5, 0.5]    # right ankle
        ])
        
        for i in range(6):
            self.target_joint_positions[i] = np.clip(
                self.target_joint_positions[i], 
                joint_limits[i][0], 
                joint_limits[i][1]
            )
        
        # Send command
        self.send_joint_command(self.target_joint_positions)
        
        # Update phase
        self.walking_phase = (self.walking_phase + 0.1) % (2 * math.pi)
        
        # Statistics
        self.step_count += 1
        
        # Log status every 50 steps (5 seconds)
        if self.step_count % 50 == 0:
            elapsed = time.time() - self.start_time
            roll, pitch, yaw = self.quaternion_to_euler(*self.current_orientation)
            self.get_logger().info(
                f'Step {self.step_count}: Time={elapsed:.1f}s, '
                f'Pitch={pitch:.3f}, Roll={roll:.3f}, '
                f'V_x={self.x_velocity:.3f}m/s'
            )


def main(args=None):
    """Main function to run the trained walker"""
    
    # Default model path
    model_path = "bipedal_walker_improved"
    
    # Check if custom model path provided
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    print("\n" + "="*60)
    print("Bipedal Walker - Running Trained Model")
    print("="*60)
    print(f"Model: {model_path}")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    rclpy.init(args=args)
    
    try:
        runner = BipedalWalkerRunner(model_path)
        
        # Wait for sensor data
        print("Waiting for sensor data...")
        start_time = time.time()
        while (not runner.imu_received or not runner.joint_received) and (time.time() - start_time < 10.0):
            rclpy.spin_once(runner, timeout_sec=0.1)
            time.sleep(0.1)
        
        if not runner.imu_received or not runner.joint_received:
            print("ERROR: Sensor data not received!")
            print(f"  IMU: {runner.imu_received}, Joints: {runner.joint_received}")
            return
        
        print("✓ Sensor data received!")
        print("✓ Starting walker...\n")
        
        # Run the walker
        rclpy.spin(runner)
        
    except KeyboardInterrupt:
        print("\n\nWalker stopped by user")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            runner.destroy_node()
        except:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()