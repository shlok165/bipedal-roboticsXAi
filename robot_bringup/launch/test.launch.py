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
    
    # Create a simple world file path (we'll create this)
    world_file_path = os.path.join(robot_bringup_pkg, 'worlds', 'bipedal_world.sdf')
    
    # --- Process URDF for Robot Description ---
    robot_description = ParameterValue(Command(['xacro ', urdf_path]), value_type=str)
    
    # --- Nodes and Actions ---
    # 1. Robot State Publisher
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}]
    )
    
    # 2. Gazebo Simulator Launch with IMU-enabled world
    gz_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_pkg, 'launch', 'gz_sim.launch.py')
        ),
        # Use our custom world file that includes the IMU plugin
        launch_arguments={
            'gz_args': f'-r {world_file_path}' if os.path.exists(world_file_path) else '-r empty.sdf',
            'on_exit_shutdown': 'true'
        }.items()
    )
    
    # 3. Spawn Robot into Gazebo
    spawn_entity_node = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'bipedal',
            '-allow_renaming', 'true',
            '-z', '1.0'  # Spawn slightly above ground
        ],
        output='screen'
    )
    
    # 4. ROS-Gazebo Bridge
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