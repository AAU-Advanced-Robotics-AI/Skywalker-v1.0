ARG ROS_DISTRO=humble
FROM osrf/ros:${ROS_DISTRO}-desktop-full

# # Example of installing programs

RUN apt-get update \
  && apt-get install -y \
  cmake \
  iputils-ping \
  cpufrequtils \
  build-essential \
  libpoco-dev \
  libeigen3-dev \
  dpkg \
  net-tools \
  can-utils \
  nano \
  python3-pip \
  liborocos-kdl-dev \
  ruby \
  tmux \
  wget \
  xorg-dev \
  python3-rosdep \
  python3-vcstool \
  python3-requests \
  ros-${ROS_DISTRO}-moveit \
  ros-${ROS_DISTRO}-controller-manager-msgs \
  #ros-${ROS_DISTRO}-gazebo-ros2-control \
  # ros-${ROS_DISTRO}-gz-ros2-control \
  ros-${ROS_DISTRO}-gazebo-ros2-control \
  ros-${ROS_DISTRO}-ros-gz \
  python3-colcon-common-extensions \
  #Libs for matplotlibcpp
  python3-dev \
  python3-matplotlib \
  #Things for GUI
  qtbase5-dev \
  qtcreator \
  && rm -rf /var/lib/apt/lists/* 

RUN gem install tmuxinator

# to put the same as the user name
ARG USERNAME=${ROS_DISTRO}
ARG USER_UID=1001 
ARG USER_GID=$USER_UID

# Creating non-root user
RUN groupadd --gid $USER_GID $USERNAME \
  && useradd -s /bin/bash --uid ${USER_UID} --gid $USER_GID -m $USERNAME \
  && mkdir /home/$USERNAME/.config && chown $USER_UID:$USER_GID /home/$USERNAME/.config


# Set up sudo (e.g. allowing to sudo nano stuff)
RUN apt-get update \
  && apt-get install -y sudo \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
  && chmod 0440 /etc/sudoers.d/$USERNAME \
  && rm -rf /var/lib/apt/lists/*

# Copy the entrypoint and bashrc scripts so we have 
# our container's environment set up correctly
COPY bashrc /home/${USERNAME}/.bashrc
