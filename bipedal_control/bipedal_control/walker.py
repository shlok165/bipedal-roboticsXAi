import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import math

class WalkerNode(Node):
    def __init__(self):
        super().__init__('walker_node')
        self.publisher_ = self.create_publisher(JointTrajectory, '/bipedal_controller/joint_trajectory', 10)
        self.timer = self.create_timer(4.0, self.publish_squat_motion) # Trigger every 4 seconds
        self.get_logger().info('WalkerNode has been started and will publish a squat motion.')

    def publish_squat_motion(self):
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = [
            'left_hip_joint', 'left_knee_joint', 
            'right_hip_joint', 'right_knee_joint'
        ]

        point_start = JointTrajectoryPoint()
        point_start.positions = [0.0, 0.0, 0.0, 0.0]
        point_start.time_from_start = Duration(sec=0, nanosec=0)
        
        point_squat = JointTrajectoryPoint()
        point_squat.positions = [-0.8, 0.8, -0.8, 0.8]
        point_squat.time_from_start = Duration(sec=2, nanosec=0)

        point_end = JointTrajectoryPoint()
        point_end.positions = [0.0, 0.0, 0.0, 0.0]
        point_end.time_from_start = Duration(sec=4, nanosec=0)
        
        trajectory_msg.points.append(point_start)
        trajectory_msg.points.append(point_squat)
        trajectory_msg.points.append(point_end)
        
        self.get_logger().info('Publishing squat motion trajectory...')
        self.publisher_.publish(trajectory_msg)

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