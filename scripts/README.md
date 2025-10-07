# SKYWALKER Project - Isaac Lab Robotics Training

A comprehensive robotics manipulation training suite built on Isaac Lab (Isaac Sim 5.0) featuring XARM7 robot configurations for assembly and point-to-point movement tasks.

## 🎯 Project Overview

**SKYWALKER** is a modular robotics training platform designed for teaching manipulation tasks using reinforcement learning. The project consists of two main environments:

1. **Original SKYWALKER Environment** - Complex assembly task with object manipulation
2. **PTP (Point-to-Point) Environment** - Simplified end-effector positioning training

## 📁 Project Structure

```
scripts/SKYWALKER/
├── README.md                           # This documentation
├── grab_skywalker/                     # Main project module
│   ├── __init__.py                     # Environment registrations
│   ├── skywalker_grab_env.py          # Base environment class
│   ├── xarm7.py                       # XARM7 robot configurations
│   ├── config/xarm7/                  # Environment configurations
│   │   ├── ptp_env_cfg.py             # 🎯 PTP environment 
│   │   ├── joint_pos_env_cfg.py       # Original SKYWALKER environment
│   │   ├── ik_abs_env_cfg.py          # IK-based control variant
│   │   └── agents/                    # Training agent configurations
│   └── mdp/                           # MDP functions (rewards, observations, etc.)
├── reach_skywalker/                   # Reaching task variants
├── skywalker_main/                    # Legacy main environments
└── [other variants]/                  # Additional project configurations
```

## 🚀 Environments

### 1. PTP (Point-to-Point) Environment 🎯

**Current Focus** - A simplified training environment for learning basic end-effector control.

#### Key Features:
- **Robot**: XARM7 7-DOF manipulator with gripper
- **Task**: Move end-effector to randomly spawned target positions
- **Episode Length**: 7 seconds (optimized for rapid training)
- **Physics**: Robot base anchored with `fix_root_link=True` for stability
- **Workspace**: Circular target spawning (200-600mm radius, ±120° angular range)
- **Training Speed**: ~35 iterations/second on CPU

#### Environment Configuration:
```python
# Environment ID
Isaac-Grab-Skywalker-PTP-PPO-v0

# Key Parameters
episode_length_s = 7.0
num_actions = 8  # 7 arm joints + 1 gripper
obs_space = 21   # joint pos/vel + ee pos + target pos + distance
```

#### Action Space:
- **arm_action** (7 dimensions): Joint position targets for XARM7
- **gripper_action** (1 dimension): Gripper open/close command

#### Observation Space:
- **joint_pos** (7): Current joint positions
- **joint_vel** (7): Current joint velocities  
- **ee_pos** (3): End-effector position
- **target_pos** (3): Target position
- **ee_to_target_dist** (1): Distance to target

#### Reward Structure:
- **ee_to_target** (weight: 10.0): Dense reward for reaching target
- **ee_target_distance** (weight: 2.0): Distance-based reward
- **action_rate_l2** (weight: -0.01): Smooth motion penalty
- **joint_vel_l2** (weight: -0.0001): Joint velocity penalty

### 2. Original SKYWALKER Environment 🏗️

**Full Assembly Task** - Complex manipulation environment with object assembly.

#### Key Features:
- **Robot**: XARM7 with surface grippers and complex object manipulation
- **Task**: Grab objects and assemble them in specific configurations
- **Episode Length**: 5 seconds
- **Objects**: Multiple manipulatable objects (cubes, cylinders, walls)
- **Complexity**: Multi-object coordination with assembly constraints

#### Environment Configuration:
```python
# Environment ID
Isaac-Grab-Skywalker-v0
Isaac-Grab-Skywalker-PPO-v0
Isaac-Grab-Skywalker-SAC-v0

# Key Features
- Surface gripper system for object manipulation
- Complex reward structure with assembly logic
- Multi-object scene with walls and obstacles
- Forward progress tracking
- Collision detection and penalties
```

#### Reward Structure (Original):
- **forward_progress**: Movement toward assembly goal
- **skywalker_reward**: Main task completion reward
- **robot_reached_goal**: Success bonus
- **simultaneous_gripper_penalty**: Gripper coordination
- **self_collision_penalty**: Safety constraints
- **wall_proximity_penalty**: Obstacle avoidance

## 🛠️ Technical Implementation

### Robot Configuration (XARM7)
```python
# Located in: grab_skywalker/xarm7.py
XARM7_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/bruno/IsaacLab/scripts/SKYWALKER/xarm7_robot.usd",
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            fix_root_link=True,  # 🔧 Critical for stability
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0.25),  # Optimized height
        joint_pos={".*": 0.0},
    ),
    actuators={"arm": ImplicitActuatorCfg(...)},
)
```

### Scene Configuration
```python
# PTP Scene Features:
- Ground plane with physics materials
- Balanced 4-light illumination system  
- Kinematic target sphere with visual markers
- Individual robot platforms (optional)
- Optimized physics settings for stability
```

### Mathematical Target Spawning
```python
# Circular workspace implementation in mdp/functions.py
def randomize_target_circular(env, env_ids, asset_cfg):
    """Mathematical polar coordinate target spawning"""
    # Radius: 200-600mm with squared distribution
    # Angle: ±120° to avoid singularities
    # Height: 35-75cm operational range
    radius_squared = torch.rand(num_envs) * (0.6**2 - 0.2**2) + 0.2**2
    radius = torch.sqrt(radius_squared)
    angle = (torch.rand(num_envs) - 0.5) * 2 * (2 * math.pi / 3)
    
    # Polar to Cartesian conversion
    x = radius * torch.cos(angle)
    y = radius * torch.sin(angle)
    z = torch.rand(num_envs) * 0.4 + 0.35
```

## 🏃‍♂️ Running the Environments

### PTP Training (Recommended)
```bash
# Standard training
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Grab-Skywalker-PTP-PPO-v0 --num_envs 16

# CPU training (single environment for debugging)
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Grab-Skywalker-PTP-PPO-v0 --num_envs 1 --device cpu

# Play mode (trained policy)
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py --task Isaac-Grab-Skywalker-PTP-Play-v0 --num_envs 1
```

### Original SKYWALKER Training
```bash
# PPO training
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Grab-Skywalker-PPO-v0 --num_envs 16

# SAC training
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Grab-Skywalker-SAC-v0 --num_envs 16
```

## 📊 Training Progress Tracking

Both environments use **Weights & Biases** for experiment tracking:

- **PTP Project**: `SKYWALKER-PTP`
- **Original Project**: `SKYWALKER` (main)

### Key Metrics:
- **Episode rewards**: Task completion progress
- **Success rate**: Target reaching frequency (PTP) / Assembly completion (Original)
- **Episode length**: Training efficiency
- **Action smoothness**: Motion quality metrics

## 🧩 Development History

### Recent Optimizations (PTP Environment):
1. **Environment Visibility Crisis** ✅ - Fixed black screen with complete scene reconstruction
2. **Robot Sliding Issues** ✅ - Resolved with `fix_root_link=True` anchoring
3. **Episode Length Optimization** ✅ - Reduced from 20s to 7s for faster training  
4. **Workspace Optimization** ✅ - Implemented circular target spawning with mathematical precision
5. **Physics Stability** ✅ - Eliminated surface gripper complexity in favor of simple anchoring

### Key Technical Decisions:
- **Surface Gripper → Fix Root Link**: Simplified physics from complex gripper system to elegant base anchoring
- **Rectangular → Circular Workspace**: Mathematical polar coordinates for realistic robot reach constraints  
- **20s → 7s Episodes**: Optimized for training speed while maintaining task complexity
- **Multi-term → Simplified Actions**: Reduced from 3 action terms to 2 (removed base_gripper_action)

## 🔧 Configuration Files

### Essential Files:
- **`ptp_env_cfg.py`**: Main PTP environment configuration
- **`ptp_functions.py`**: MDP functions including circular target spawning
- **`skrl_ptp_ppo_cfg.yaml`**: PPO training hyperparameters
- **`xarm7.py`**: Robot articulation definitions

### Key Configuration Classes:
```python
@configclass
class PTPEnvCfg(ManagerBasedRLEnvCfg):
    """Point-to-Point training environment"""
    scene: PTPSceneCfg = PTPSceneCfg(num_envs=4096, env_spacing=2.0)
    episode_length_s = 7.0
    decimation = 2
    
@configclass  
class PTPEnvCfg_PLAY(PTPEnvCfg):
    """Play mode configuration"""
    scene: PTPSceneCfg = PTPSceneCfg(num_envs=50, env_spacing=2.5)
```

## 🎓 Learning Outcomes

### PTP Environment Learning Goals:
- **Basic End-Effector Control**: Fundamental manipulation skills
- **Workspace Awareness**: Understanding robot reach limitations
- **Smooth Motion Planning**: Efficient trajectory generation
- **Target Reaching**: Precision positioning tasks

### Original SKYWALKER Learning Goals:
- **Complex Assembly Tasks**: Multi-object coordination
- **Gripper Control**: Surface attachment and object manipulation
- **Spatial Reasoning**: 3D assembly planning
- **Constraint Satisfaction**: Working within physical limitations

## 🚨 Troubleshooting

### Common Issues:

1. **Robot Sliding**: Ensure `fix_root_link=True` in articulation properties
2. **Black Screen**: Verify complete scene configuration with ground plane and lighting
3. **Training Slow**: Consider reducing `num_envs` or using `--device cpu` for debugging
4. **Target Unreachable**: Check circular spawning parameters in `randomize_target_circular`

### Debug Commands:
```bash
# Check environment registration
python -c "import gymnasium as gym; print([env for env in gym.envs.registry.env_specs.keys() if 'Skywalker' in env])"

# Validate configuration
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Grab-Skywalker-PTP-PPO-v0 --num_envs 1 --headless
```

## 📈 Performance Benchmarks

### PTP Environment:
- **Training Speed**: ~35 it/s (CPU), ~200+ it/s (GPU)
- **Convergence**: Typically 10k-50k steps for basic reaching
- **Success Rate**: >90% target reaching after convergence
- **Workspace Coverage**: Full 600mm circular workspace

### Original SKYWALKER:
- **Training Speed**: ~25 it/s (CPU), ~150+ it/s (GPU)  
- **Convergence**: 100k+ steps for assembly tasks
- **Success Rate**: Variable based on assembly complexity
- **Task Complexity**: Multi-object coordination challenges

## 📚 References

- **Isaac Lab Documentation**: [https://isaac-sim.github.io/IsaacLab/](https://isaac-sim.github.io/IsaacLab/)
- **XARM7 Robot**: 7-DOF manipulator with gripper end-effector
- **SKRL Library**: Reinforcement learning framework for Isaac Lab
- **Weights & Biases**: Experiment tracking and visualization

---

## 🤝 Contributing

To extend or modify the SKYWALKER environments:

1. **Environment Variants**: Create new config files in `config/xarm7/`
2. **MDP Functions**: Add custom rewards/observations in `mdp/`
3. **Robot Configurations**: Modify `xarm7.py` for hardware changes
4. **Training Agents**: Update agent configs in `config/xarm7/agents/`

**Happy Training!** 🤖✨
