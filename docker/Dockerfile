ARG JETPACK_VERSION=6

FROM ultralytics/ultralytics:latest-jetson-jetpack${JETPACK_VERSION}

ARG ROS_EDITION=humble
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV WORKSPACE_DIR=/skeleton_ws


RUN apt update  && \
    apt install -y git && \
    apt install -y wget && \
    apt install -y locales    && \    
    locale-gen en_US en_US.UTF-8   && \    
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8   && \    
    export LANG=en_US.UTF-8   && \    
    apt install -y software-properties-common   && \    
    add-apt-repository universe   && \    
    apt update && apt install curl -y   && \    
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg   && \    
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null   && \    
    apt update   && \    
    apt install -y tzdata  && \    
    ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime  && \    
    dpkg-reconfigure -f noninteractive tzdata  && \
    if [ "$ROS_EDITION" = "foxy" ]; then \
            apt install -y ros-foxy-ros-base python3-argcomplete; \
            # Action A
        elif [ "$ROS_EDITION" = "humble" ]; then \
            apt install -y ros-humble-ros-base; \
            # Action B
        else \
            exit 0 ; \
            # Default action
        fi && \    
    apt install -y python3-colcon-common-extensions   && \    
    apt install -y libboost-all-dev && \
    rm -rf /var/lib/apt/lists/*

# Change to root directory to ensure correct placement of the workspace
WORKDIR /${WORKSPACE_DIR}/src

# Clone the person_skeleton_gestures repository into the desired workspace directory
RUN rm -rf * && \
    git clone -b jetpack6 https://github.com/Cogniteam-ACIIST/person_skeleton_gestures.git && \
    git clone -b $ROS_EDITION https://github.com/ros-perception/vision_opencv.git

WORKDIR ${WORKSPACE_DIR}/src/person_skeleton_gestures/skeleton_gesture_detector/resource
# Download the YOLOv7 model file
RUN wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt

# Set the working directory to the workspace
WORKDIR /${WORKSPACE_DIR}

RUN python3 -m pip install setuptools==70.0.0

# Build the workspace
RUN . /opt/ros/$ROS_EDITION/setup.sh && colcon build --symlink-install

COPY entrypoint.sh /entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /entrypoint.sh

# Set the entrypoint script to launch your ROS 2 application
ENTRYPOINT ["/entrypoint.sh"]
