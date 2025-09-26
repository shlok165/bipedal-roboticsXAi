import os
from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():

    # --- Get Package and File Paths ---
    robot_description_pkg = get_package_share_path('robot_description')
    robot_bringup_pkg = get_package_share_path('robot_bringup')
    ros_gz_sim_pkg = get_package_share_path('ros_gz_sim')

    urdf_path = os.path.join(robot_description_pkg, 'urdf', 'bipedal.urdf.xacro')
    rviz_config_path = os.path.join(robot_bringup_pkg, 'config', 'rviz.rviz')
    
    # --- Process URDF for Robot Description ---
    robot_description = ParameterValue(Command(['xacro ', urdf_path]), value_type=str)

    # --- Nodes and Actions ---

    # 1. Robot State Publisher
    # Takes the URDF and joint states and publishes the TF transforms for all links.
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}]
    )

    # 2. Gazebo Simulator Launch
    # This is the correct way: including the launch file provided by the package.
    gz_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_pkg, 'launch', 'gz_sim.launch.py')
        ),
        # Pass the '-r' flag to run the simulation and specify the world file.
        launch_arguments={'gz_args': '-r empty.sdf'}.items()
    )

    # 3. Spawn Robot into Gazebo
    # Runs the 'create' executable to spawn a robot from the 'robot_description' topic.
    spawn_entity_node = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'bipedal',
            '-allow_renaming', 'true'
        ],
        output='screen'
    )
    
    # 4. ROS-Gazebo Bridge
    # Connects ROS 2 topics with Gazebo's transport layer.
    # We bridge the clock for simulation time and TF for Gazebo-related transforms.
    bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            # Bridge the clock
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            
            # Bridge the IMU data from Gazebo topic to ROS 2 topic
            '/bipedal/imu@sensor_msgs/msg/Imu[gz.msgs.IMU'
        ],
        # We need to remap the frame_id from the Gazebo default to our robot's frame
        remappings=[
            ('/bipedal/imu', '/bipedal/imu_remapped')
        ],
        output='screen'
    )

    # 5. Controller Spawners
    # The gz_ros2_control plugin in your URDF starts the controller_manager.
    # These spawners then load your controllers into it.
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
    )

    bipedal_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["bipedal_controller", "--controller-manager", "/controller_manager"],
    )

    # 6. RViz2
    rviz2_node = Node(
        package="rviz2",
        executable="rviz2",
        arguments=['-d', rviz_config_path]
    )

    # --- Return the Launch Description ---
    return LaunchDescription([
        gz_sim_launch,
        bridge_node,
        robot_state_publisher_node,
        spawn_entity_node,
        joint_state_broadcaster_spawner,
        bipedal_controller_spawner,
        rviz2_node
    ])