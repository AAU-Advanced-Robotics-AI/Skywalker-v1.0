"Phase three Rewards "






def robot_base_to_goal_distance(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal_marker"),
) -> torch.Tensor:
    """Compute reward based on distance from robot base to per-env goal marker."""
    root_pos = env.scene[robot_cfg.name].data.root_pos_w[:, :2]  # shape (num_envs, 2)
    goal_pos = env.scene[goal_cfg.name].data.root_pos_w[:, :2]   # shape (num_envs, 2)

    distance = torch.norm(root_pos - goal_pos, dim=1)

    # Optional: print first 30 distances
    #print("Distances (first 30 envs):", distance[:30].cpu().numpy())

    return 1 - torch.tanh(distance)


def robot_base_to_goal_distance_fine(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal_marker"),
) -> torch.Tensor:
    """Compute reward based on distance from robot base to per-env goal marker."""
    root_pos = env.scene[robot_cfg.name].data.root_pos_w[:, :2]  # shape (num_envs, 2)
    goal_pos = env.scene[goal_cfg.name].data.root_pos_w[:, :2]   # shape (num_envs, 2)

    distance = torch.norm(root_pos - goal_pos, dim=1)

    # Optional: print first 30 distances
    #print("Distances (first 30 envs):", distance[:30].cpu().numpy())

    return 1 - torch.tanh(distance / 0.1)



def bad_hold_penalty(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube1"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal_marker"),
    grip_term: str = "gripper_action",
    lam: float = 0.02,
) -> torch.Tensor:
    """Penalise still holding *cube1* once the base has passed it w.r.t. the goal."""
    root = env.scene[robot_cfg.name].data.root_pos_w[:, :2]
    goal = env.scene[goal_cfg.name].data.root_pos_w[:, :2]
    cube = env.scene[cube_cfg.name].data.target_pos_w[..., 0, :2]

    d_goal = torch.norm(root - goal, dim=1)
    d_cube_goal = torch.norm(cube - goal, dim=1)

    holding = is_grasping_fixed_object(env, cube_cfg=cube_cfg, grip_term=grip_term)
    return -lam * holding.squeeze(1) * (d_goal < d_cube_goal)






def robot_goal_docking_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_cfg:  SceneEntityCfg = SceneEntityCfg("goal_marker"),
    grip_term: str = "gripper_action2",
    std: float = 0.25,           # distance at which reward ≈ e-1 ≈ 0.37
    tol: float = 0.12,           # “close-enough” zone (same units as distance)
    close_bonus: float = 1.0,    # extra bump for closing inside tol
    far_penalty: float = -0.3,   # small negative for closing too early
) -> torch.Tensor:
    """
    Dense docking reward in robot-base frame.

    • Always gives exp(-d / std) so the critic sees a gradient.
    • Adds +close_bonus when gripper2 *closes* within tol of the goal.
    • Adds –far_penalty when gripper2 closes farther than tol.
    """
    robot = env.scene[robot_cfg.name]
    goal  = env.scene[goal_cfg.name]
    gripper = env.action_manager.get_term(grip_term)

    # distance in XY plane (robot frame == cylinder frame)
    root_pos = robot.data.root_pos_w[:, :2]
    goal_pos = goal.data.root_pos_w[:, :2]
    dist = torch.norm(root_pos - goal_pos, dim=1)        # (N,)

    base_reward = torch.exp(-dist / std)                 # (0, 1]

    closed = gripper.is_closed().float()                 # (N,)
    near   = (dist < tol).float()

    # combine signals
    shaped = base_reward + close_bonus * closed * near + far_penalty * closed * (1 - near)
    return shaped














##---ReTerms

    # grasp_cube1_weighted = RewTerm(
    # func   = mdp.distance_weighted_grasp,
    # weight = 1.0,
    # params = {
    #     "cube_cfg": SceneEntityCfg("cube1"),
    #     "grip_term": "gripper_action",   # or whatever term controls cube 1 gripper
    # },
    # )
    # grasp_cube2_weighted = RewTerm(
    #     func   = mdp.distance_weighted_grasp,
    #     weight = 1.0,
    #     params = {
    #         "cube_cfg": SceneEntityCfg("cube2"),
    #         "grip_term": "gripper_action",
    #     },
    # )

    # approach_cube1 = RewTerm(              # NEW
    #     func   = mdp.object_ee_distance,
    #     weight = 3.0,
    #     params = {"cube_cfg": SceneEntityCfg("cube1"), "std": 0.20},
    # )

    # grab_cube1 = RewTerm(
    #     func=mdp.is_grasping_fixed_object,
    #     weight=5,
    #     params={"cube_cfg": SceneEntityCfg("cube1")},
    # )


    # approach_cube2 = RewTerm(              # NEW
    #     func   = mdp.object_ee_distance,
    #     weight = 3.0,
    #     params = {"cube_cfg": SceneEntityCfg("cube2"), "std": 0.20},
    # )

    # grab_cube2 = RewTerm(
    #     func=mdp.is_grasping_fixed_object,
    #     weight=5,
    #     params={"cube_cfg": SceneEntityCfg("cube2")},
    # )


    # hold_far_cube_penalty = RewTerm(
    #     func   = mdp.hold_far_cube_penalty,
    #     weight = 1.0,          # λ lives inside helper; scale here if needed
    #     params = {
    #         "reach_thresh": 0.50,   # comfortable reach (tweak per arm)
    #         "lam": 1.0,
    #     },
    #     )



    # robot_goal_tracking = RewTerm(
    #     func=mdp.robot_base_to_goal_distance,
    #     weight=0.2,
    #     params={}, 
    # )

    # robot_goal_tracking_fine = RewTerm(
    #     func=mdp.robot_base_to_goal_distance_fine,
    #     weight=1.0,
    #     params={}, 
    # )

    # reaching_object = RewTerm(
    #     func=mdp.object_ee_distance,
    #     weight=5,
    #     params={"std": 0.1},
    # )

    # ee_alignment = RewTerm(
    #     func=mdp.ee_cube_orientation_alignment,
    #     weight=1.5,
    #     params={}
    # )

    # ee_approach_alignment = RewTerm(
    #     func=mdp.ee_approach_alignment_in_base,
    #     weight=1.2,
    #     params={}
    # )



    # sim_grab_penalty = RewTerm(
    #     func=mdp.simultaneous_gripper_penalty,
    #     weight=0.5,
    #     params={}
    # )

    # cylinder_to_goal = RewTerm(
    # func   = mdp.robot_goal_docking_reward,
    # weight = 5.0,
    # params = {}          # use defaults above; tune later if needed
    # )

    

    # # grab_floor = RewTerm(
    # #     func=mdp.is_gripper2_closed_around_goal,
    # #     weight=0.0,
    # #     params={"tol": 0.15}
    # # )

    # action_rate = RewTerm(
    #     func=mdp.action_rate_l2,
    #     weight=1e-5,
    #     params={}
    # )

    # joint_vel = RewTerm(
    #     func=mdp.joint_vel_l2,
    #     weight=1e-5,
    #     params={"asset_cfg": SceneEntityCfg("robot")}
    # )

    # self_collision = RewTerm(
    #     func=mdp.self_collision_penalty,
    #     weight=1.0,
    #     params={"asset_cfg": SceneEntityCfg("robot")}
    # )

##-------CurriculumCfg
    # self_grasp = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "self_grasp", "weight": 2.0, "num_steps": 1000}
    # )



    # reaching_object = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "reaching_object", "weight": 1.5, "num_steps": 1000}
    # )

    # robot_goal_tracking = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "robot_goal_tracking", "weight": 1.5, "num_steps": 1000}
    # )

    # robot_goal_tracking = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "robot_goal_tracking_fine", "weight": 25.0, "num_steps": 1000}
    # )

    # grab_floor = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "grab_floor", "weight": 20.0, "num_steps": 5000}
    # )

    # sim_grab_penalty = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "sim_grab_penalty", "weight": 1.0, "num_steps": 2000}
    # )


    # grab_cube = CurrTerm(
    #     func=mdp.modify_reward_weight,
    #     params={"term_name": "grab_cube", "weight": 1.5, "num_steps": 1500}
    # )


