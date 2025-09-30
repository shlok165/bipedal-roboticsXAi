import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

class WalkerNode(Node):
    def __init__(self):
        super().__init__('walker_node')
        self.publisher_ = self.create_publisher(JointTrajectory, '/bipedal_controller/joint_trajectory', 10)
        
        # Create a timer to repeatedly publish the walking motion
        self.timer = self.create_timer(10.0, self.publish_walk_motion)
        
        self.get_logger().info('WalkerNode has been started.')
        
        # Publish the first walk cycle immediately on startup
        self.publish_walk_motion()

    def publish_walk_motion(self):
        self.get_logger().info('Publishing hardcoded slow walk trajectory...')
        
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint'
        ]
        
        # Standing position
        p0 = JointTrajectoryPoint()
        p0.positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        p0.time_from_start = Duration(sec=0, nanosec=0)
        
        # Lean forward slightly (POSITIVE hip = forward lean)
        p1 = JointTrajectoryPoint()
        p1.positions = [0.1, 0.1, -0.1, 0.1, 0.1, -0.1]
        p1.time_from_start = Duration(sec=1, nanosec=0)
        
        # Lift right leg (bend knee)
        p2 = JointTrajectoryPoint()
        p2.positions = [0.1, 0.1, -0.1, 0.1, 0.5, -0.2]
        p2.time_from_start = Duration(sec=2, nanosec=0)
        
        # Swing right leg forward
        p3 = JointTrajectoryPoint()
        p3.positions = [-0.1, 0.1, 0.0, 0.3, 0.5, -0.3]
        p3.time_from_start = Duration(sec=3, nanosec=0)
        
        # Plant right leg, straighten knee
        p4 = JointTrajectoryPoint()
        p4.positions = [-0.1, 0.1, 0.0, 0.3, 0.1, -0.2]
        p4.time_from_start = Duration(sec=4, nanosec=0)
        
        # Lift left leg (bend knee)
        p5 = JointTrajectoryPoint()
        p5.positions = [-0.1, 0.5, -0.2, 0.1, 0.1, -0.1]
        p5.time_from_start = Duration(sec=5, nanosec=0)
        
        # Swing left leg forward
        p6 = JointTrajectoryPoint()
        p6.positions = [0.3, 0.5, -0.3, -0.1, 0.1, 0.0]
        p6.time_from_start = Duration(sec=6, nanosec=0)
        
        # Plant left leg, straighten knee
        p7 = JointTrajectoryPoint()
        p7.positions = [0.3, 0.1, -0.2, -0.1, 0.1, 0.0]
        p7.time_from_start = Duration(sec=7, nanosec=0)
        
        # Return to center standing
        p8 = JointTrajectoryPoint()
        p8.positions = [0.1, 0.1, -0.1, 0.1, 0.1, -0.1]
        p8.time_from_start = Duration(sec=8, nanosec=0)
        
        # Final standing position
        p9 = JointTrajectoryPoint()
        p9.positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        p9.time_from_start = Duration(sec=9, nanosec=0)
        
        trajectory_msg.points.append(p0)
        trajectory_msg.points.append(p1)
        trajectory_msg.points.append(p2)
        trajectory_msg.points.append(p3)
        trajectory_msg.points.append(p4)
        trajectory_msg.points.append(p5)
        trajectory_msg.points.append(p6)
        trajectory_msg.points.append(p7)
        trajectory_msg.points.append(p8)
        trajectory_msg.points.append(p9)
        
        self.publisher_.publish(trajectory_msg)
        self.get_logger().info('Trajectory published.')

def main(args=None):
    rclpy.init(args=args)
    node = WalkerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()