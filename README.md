# P10_MT_RL_SKYWALKER
This repo is for Master thesis project on Reinforcement learning in the space industry



# Startup and installation

When starting up this Repo the first time, the docker has to be build and running before the docker can be used. This is dependent on the system you are using.

***FIRST*** thing to do, is to add your user in the [bashrc](bashrc) file in the buttom line.


## Ubuntu

Start by building the docker by using the [**docker_build.sh**](docker_build.sh) script. This will take some time.

```cmd
sh docker_build.sh
```

The run the docker using the [**docker.sh**](docker.sh) script. This will open the docker and change the user in the cmd to  ***humble@docker-desktop:/ros2_ws***


```cmd
sh docker.sh
```

## Windows
For windows, we first need to install Docker Desktop, which can be downloaded [here](https://docs.docker.com/desktop/setup/install/windows-install/)

After Docker Desktop is installed, the [**docker_build.ps1**](docker_build.ps1) script can be run. This will take some time, and also open the Docker Desktop if it is not already running.
```cmd
.\docker_build.ps1
```

After the docker has benn made, the docker can be started and run using the [**docker.ps1**](docker.ps1) script.  This will open the docker and change the user in the cmd to  ***humble@docker-desktop:/ros2_ws***







