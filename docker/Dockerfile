# Use the base image
FROM cognimbus/yolo_detector:test

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV WORKSPACE_DIR=/skeleton_ws

# Install necessary tools
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

#RUN apt-get update && apt install ros-humble-cv-bridge -y

# Change to root directory to ensure correct placement of the workspace
WORKDIR /

# Clone the person_skeleton_gestures repository into the desired workspace directory
RUN mkdir -p ${WORKSPACE_DIR}/src && git clone https://github.com/Cogniteam-ACIIST/person_skeleton_gestures.git ${WORKSPACE_DIR}/src/person_skeleton_gestures

# Download the YOLOv7 model file
RUN wget -P ${WORKSPACE_DIR}/src/person_skeleton_gestures/skeleton_gesture_detector/resource \
    https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt

# Set the working directory to the workspace
WORKDIR ${WORKSPACE_DIR}

# Build the workspace
RUN /bin/bash -c "source /opt/ros/foxy/setup.bash && colcon build --symlink-install"

COPY entrypoint.sh /entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /entrypoint.sh

# Set the entrypoint script to launch your ROS 2 application
ENTRYPOINT ["/entrypoint.sh"]
