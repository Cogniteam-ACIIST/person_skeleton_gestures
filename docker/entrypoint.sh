#!/bin/bash
# Source the ROS 2 Humble setup
source /opt/ros/foxy/setup.bash

# Source the workspace setup (if it exists)
if [ -f /skeleton_ws/install/setup.bash ]; then
    source /skeleton_ws/install/setup.bash
fi

# Execute the command passed to the container
exec "$@"