# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent with full physics data recording.

Usage:
    python my_play.py --task Go2-Flat-v0 --checkpoint /path/to/model.pt --headless --record_physics --num_envs 1 --cmd_sequence "2,0,0,0;5,0.5,0,0;2,0,0,0"

    python my_play.py --task Go2-Flat-v0 --checkpoint /path/to/model.pt --record_physics --num_envs 1 --keyboard

    python my_play.py --task Go2-Stairs-v0 --checkpoint /path/to/model.pt --headless --record_physics --num_envs 1 --cmd_sequence "2,0,0,0;8,0.3,0,0;2,0,0,0" --step_height 0.1,0.3
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import time
import csv
import json
import numpy as np

from isaaclab.app import AppLauncher

import cli_args

parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL and record full physics data.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None, help="Task name, e.g. Go2-Flat-v0, A1-Stairs-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--use_pretrained_checkpoint", action="store_true")
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--keyboard", action="store_true", default=False)
parser.add_argument("--record_physics", action="store_true", default=False)
parser.add_argument("--physics_save_path", type=str, default=None)
parser.add_argument("--record_interval", type=int, default=1)

parser.add_argument("--cmd_sequence", type=str, default=None,
                    help="Command sequence: 'dur1,vx1,vy1,wz1;dur2,vx2,vy2,wz2;...'")

parser.add_argument("--reset_pos", type=str, default=None,
                    help="Spawn position: 'x,y,z,yaw'")

parser.add_argument("--step_height", type=str, default=None,
                    help="Stairs step height range: 'min,max' (default: 0.05,0.23)")
parser.add_argument("--step_width", type=float, default=None,
                    help="Stairs step width (default: 0.3)")
parser.add_argument("--platform_width", type=float, default=None,
                    help="Platform width for stairs/slope (default: stairs=3.0, slope=2.0)")
parser.add_argument("--slope_range", type=str, default=None,
                    help="Slope range: 'min,max' (default: 0.0,0.4)")
parser.add_argument("--noise_range", type=str, default=None,
                    help="Rough terrain noise range: 'min,max' (default: 0.02,0.10)")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl_utils import camera_follow


# ==============================================================================
# 环境注册
# ==============================================================================

ROBOT_CONFIG_MAP = {
    "go2": "robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.unitree_go2.rough_env_cfg:UnitreeGo2RoughEnvCfg",
}

TERRAIN_TYPES = ["flat", "stairs", "rough", "slope"]


def _parse_task_name(task_name: str) -> tuple[str, str]:
    parts = task_name.split("-")
    if len(parts) < 2:
        raise ValueError(f"Task name must be 'Robot-Terrain-v0', got: {task_name}")
    robot = parts[0].lower()
    terrain = parts[1].lower()
    if terrain not in TERRAIN_TYPES:
        raise ValueError(f"Unknown terrain type: '{terrain}'. Available: {TERRAIN_TYPES}")
    if robot not in ROBOT_CONFIG_MAP:
        raise ValueError(f"Unknown robot: '{robot}'. Available: {list(ROBOT_CONFIG_MAP.keys())}")
    return robot, terrain


def _register_env(task_name: str):
    if task_name in gym.envs.registry:
        return
    robot, _ = _parse_task_name(task_name)
    gym.register(
        id=task_name,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": ROBOT_CONFIG_MAP[robot],
            "rsl_rl_cfg_entry_point": f"robot_lab.tasks.manager_based.locomotion.velocity.config.quadruped.unitree_{robot}.agents.rsl_rl_ppo_cfg:Unitree{robot.capitalize()}RoughPPORunnerCfg",
        },
    )
    print(f"[INFO] 自动注册环境: {task_name}")


# ==============================================================================
# 地形配置
# ==============================================================================

def _parse_range(arg: str | None, default: tuple) -> tuple:
    if arg is None:
        return default
    parts = arg.split(",")
    if len(parts) != 2:
        raise ValueError(f"Range must be 'min,max', got: {arg}")
    return (float(parts[0]), float(parts[1]))


def _create_terrain_cfg(terrain_type: str):
    import isaaclab.terrains as terrain_gen
    from isaaclab.terrains import TerrainGeneratorCfg

    if terrain_type == "flat":
        print("[INFO] 地形: 纯平坦 (plane)")
        return None

    platform_width = args_cli.platform_width

    if terrain_type == "stairs":
        step_height = _parse_range(args_cli.step_height, (0.05, 0.23))
        step_width = args_cli.step_width if args_cli.step_width is not None else 0.3
        platform = platform_width if platform_width is not None else 3.0
        return TerrainGeneratorCfg(
            size=(20.0, 20.0), num_rows=1, num_cols=1, curriculum=False,
            sub_terrains={
                "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                    proportion=1.0, step_height_range=step_height,
                    step_width=step_width, platform_width=platform,
                    border_width=1.0, holes=False,
                ),
            },
        )

    if terrain_type == "rough":
        noise_range = _parse_range(args_cli.noise_range, (0.02, 0.10))
        print(f"[INFO] 地形: 纯粗糙 noise_range={noise_range}")
        return TerrainGeneratorCfg(
            size=(12.0, 12.0), num_rows=1, num_cols=1, curriculum=False,
            sub_terrains={
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=1.0, noise_range=noise_range,
                    noise_step=0.02, border_width=0.25,
                ),
            },
        )

    if terrain_type == "slope":
        slope_range = _parse_range(args_cli.slope_range, (0.0, 0.4))
        platform = platform_width if platform_width is not None else 2.0
        return TerrainGeneratorCfg(
            size=(20.0, 20.0), num_rows=1, num_cols=1, curriculum=False,
            sub_terrains={
                "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                    proportion=1.0, slope_range=slope_range,
                    platform_width=platform, border_width=0.25,
                ),
            },
        )

    raise ValueError(f"Unknown terrain type: {terrain_type}")


# ==============================================================================
# 命令序列
# ==============================================================================

def parse_cmd_sequence(sequence_str: str):
    boundaries = []
    t_sum = 0.0
    for seg in sequence_str.split(";"):
        parts = seg.split(",")
        if len(parts) != 4:
            raise ValueError(f"Command segment must be 'dur,vx,vy,wz', got: {seg}")
        dur, vx, vy, wz = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
        t_sum += dur
        boundaries.append((t_sum, vx, vy, wz))
    print(f"[INFO] 命令序列: 总时长 {t_sum:.1f}s, 共 {len(boundaries)} 段")
    for i, (end_t, vx, vy, wz) in enumerate(boundaries):
        print(f"       [{i}] 0-{end_t:.1f}s: lin_x={vx}, lin_y={vy}, ang_z={wz}")
    return boundaries, t_sum


def make_cmd_getter(boundaries, device):
    def get_cmd(t):
        for end_t, vx, vy, wz in boundaries:
            if t < end_t:
                return torch.tensor([[vx, vy, wz]], device=device)
        _, vx, vy, wz = boundaries[-1]
        return torch.tensor([[vx, vy, wz]], device=device)
    return get_cmd


# ==============================================================================
# 字段含义
# ==============================================================================

def build_field_descriptions(robot, foot_names):
    joint_names = [name for name in robot.joint_names]
    body_names = [name for name in robot.data.body_names]
    rows = []

    def add_section(title):
        rows.append([f"=== {title} ===", "", ""])

    def add_field(name, description, unit="", mapping=""):
        rows.append([name, description, unit, mapping])

    add_section("基础时间")
    add_field("time", "仿真时间 (sim_step * dt)", "秒 (s)")
    add_field("record_step", "记录步数序号", "")

    add_section("命令 (base_velocity)")
    add_field("cmd_0", "目标线速度 X", "m/s")
    add_field("cmd_1", "目标线速度 Y", "m/s")
    add_field("cmd_2", "目标角速度 Z", "rad/s")

    add_section("基座位姿")
    add_field("pos_z", "基座 Z 轴位置", "m")
    add_field("quat_w", "基座姿态四元数 W", "", "[w,x,y,z]")
    add_field("quat_x", "基座姿态四元数 X", "")
    add_field("quat_y", "基座姿态四元数 Y", "")
    add_field("quat_z", "基座姿态四元数 Z", "")

    add_section("基座速度")
    for axis, name in zip(["X", "Y", "Z"], ["lin_x", "lin_y", "lin_z"]):
        add_field(name, f"基座线速度 {axis}", "m/s")
    for axis, name in zip(["X", "Y", "Z"], ["ang_x", "ang_y", "ang_z"]):
        add_field(name, f"基座角速度 {axis}", "rad/s")

    add_section(f"动作 (策略输出, 共 {len(joint_names)} 个关节)")
    for i, name in enumerate(joint_names):
        add_field(f"action_{i}", f"关节 {name} 目标位置偏移", "rad")

    add_section(f"关节状态 (按 {joint_names} 顺序)")
    for i, name in enumerate(joint_names):
        add_field(f"joint_pos_{i}", f"关节 {name} 位置", "rad")
    for i, name in enumerate(joint_names):
        add_field(f"joint_vel_{i}", f"关节 {name} 速度", "rad/s")
    for i, name in enumerate(joint_names):
        add_field(f"applied_torque_{i}", f"关节 {name} 施加力矩", "Nm")
    for i, name in enumerate(joint_names):
        add_field(f"joint_power_{i}", f"关节 {name} 功率", "W")
    for i, name in enumerate(joint_names):
        add_field(f"joint_limit_violation_{i}", f"关节 {name} 限位标志", "")

    add_section(f"接触力 (按 {body_names} 顺序)")
    for i, name in enumerate(body_names):
        for axis, offset in zip(["X", "Y", "Z"], [0, 1, 2]):
            add_field(f"net_force_{i*3+offset}", f"刚体 {name} 净接触力 {axis}", "N")

    add_section("接触点位置")
    for i, name in enumerate(body_names):
        for axis, offset in zip(["X", "Y", "Z"], [0, 1, 2]):
            add_field(f"contact_pos_{i*3+offset}", f"刚体 {name} 接触点 {axis}", "m")

    add_section("腾空/接触时间")
    for i, name in enumerate(body_names):
        add_field(f"current_air_time_{i}", f"刚体 {name} 当前腾空时间", "s")
    for i, name in enumerate(body_names):
        add_field(f"last_air_time_{i}", f"刚体 {name} 上次腾空时间", "s")
    for i, name in enumerate(body_names):
        add_field(f"current_contact_time_{i}", f"刚体 {name} 当前接触时间", "s")

    add_section("基座扩展")
    for i, axis in enumerate(["X", "Y", "Z"]):
        add_field(f"root_com_pos_{i}", f"基座质心位置 {axis}", "m")
    for i, axis in enumerate(["X", "Y", "Z"]):
        add_field(f"projected_gravity_{i}", f"重力投影 {axis}", "m/s²")
    add_field("heading_w", "基座偏航角", "rad")

    add_section("高度扫描")
    add_field("height_scan_*", "高度扫描器射线击中点", "m")

    add_section(f"足端数据 (按 {foot_names} 顺序)")
    for fn in foot_names:
        for axis in ["X", "Y", "Z"]:
            add_field(f"foot_{fn}_pos_{axis.lower()}", f"足端 {fn} 位置 {axis}", "m")
    for fn in foot_names:
        for axis in ["X", "Y", "Z"]:
            add_field(f"foot_{fn}_vel_{axis.lower()}", f"足端 {fn} 线速度 {axis}", "m/s")
    for fn in foot_names:
        add_field(f"foot_height_{fn}", f"足端 {fn} 相对基座高度", "m")

    add_section("汇总")
    add_field("total_contact_force_magnitude", "所有刚体净接触力大小之和", "N")

    return rows


# ==============================================================================
# 主函数
# ==============================================================================

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    if args_cli.keyboard and args_cli.cmd_sequence:
        raise ValueError("--keyboard and --cmd_sequence are mutually exclusive.")

    task_name = args_cli.task
    robot_type, terrain_type = _parse_task_name(task_name)
    terrain_cfg = _create_terrain_cfg(terrain_type)

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 64
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env_cfg.scene.terrain.terrain_generator = terrain_cfg
    if terrain_cfg is None:
        env_cfg.scene.terrain.terrain_type = "plane"
    else:
        env_cfg.scene.terrain.terrain_type = "generator"
        env_cfg.scene.terrain.max_init_terrain_level = None

    env_cfg.scene.robot.activate_contact_sensors = True
    env_cfg.sim.enable_scene_query_support = True
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.command_levels_ang_vel = None
    env_cfg.curriculum.terrain_levels = None

    if args_cli.reset_pos:
        parts = args_cli.reset_pos.split(",")
        if len(parts) != 4:
            raise ValueError(f"--reset_pos must be 'x,y,z,yaw', got: {args_cli.reset_pos}")
        x, y, z, yaw = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
        env_cfg.events.randomize_reset_base.params = {
            "pose_range": {"x": (x, x), "y": (y, y), "z": (z, z),
                           "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (yaw, yaw)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                               "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
        }
        print(f"[INFO] 固定出生位置: x={x}, y={y}, z={z}, yaw={yaw}")

    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        env_cfg.terminations.terrain_out_of_bounds = None
        config = Se2KeyboardCfg(
            v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1],
            v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
            omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
        )
        controller = Se2Keyboard(config)
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device),
        )

    cmd_boundaries = None
    total_duration = None
    if args_cli.cmd_sequence:
        cmd_boundaries, total_duration = parse_cmd_sequence(args_cli.cmd_sequence)
        env_cfg.episode_length_s = total_duration + 2.0

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] No pre-trained checkpoint available for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir
    from isaaclab.sim import RigidBodyMaterialCfg

    if terrain_type == "slope":
        env_cfg.scene.terrain.physics_material = RigidBodyMaterialCfg(
            static_friction=1.5,
            dynamic_friction=1.2,
        )

    env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if cmd_boundaries is not None:
        get_cmd_fn = make_cmd_getter(cmd_boundaries, env.unwrapped.device)
        if hasattr(env.unwrapped, "command_manager"):
            orig_get_command = env.unwrapped.command_manager.get_command
            def fixed_get_command(name):
                if name == "base_velocity":
                    t = env.unwrapped.episode_length_buf[0].item() * env.unwrapped.step_dt
                    return get_cmd_fn(t)
                return orig_get_command(name)
            env.unwrapped.command_manager.get_command = fixed_get_command
            print("[INFO] 已注入固定命令序列")

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=None, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=None, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    csv_file = None
    csv_writer = None
    csv_file_path = None
    track_count = 0

    if args_cli.record_physics:
        timestamp = os.path.basename(os.path.dirname(resume_path))
        ckpt_name = os.path.splitext(os.path.basename(resume_path))[0]
        if args_cli.physics_save_path:
            base, ext = os.path.splitext(args_cli.physics_save_path)
            csv_file_path = f"{base}_{timestamp}{ext if ext else '.csv'}"
            os.makedirs(os.path.dirname(csv_file_path) or ".", exist_ok=True)
        else:
            csv_file_path = os.path.join(log_dir, f"{task_name}_{ckpt_name}_{terrain_type}_{timestamp}.csv")
        print(f"[INFO] 数据将实时写入: {csv_file_path}")

    robot = env.unwrapped.scene["robot"]
    contact_sensor = env.unwrapped.scene["contact_forces"] if "contact_forces" in env.unwrapped.scene.keys() else None

    metadata = {}
    metadata["joint_names"] = [name for name in robot.joint_names]
    limits = robot.data.joint_pos_limits[0].detach().cpu().numpy()
    metadata["joint_limits_lower"] = [float(limits[i][0]) for i in range(len(limits))]
    metadata["joint_limits_upper"] = [float(limits[i][1]) for i in range(len(limits))]
    default_pos = robot.data.default_joint_pos[0].detach().cpu().numpy()
    metadata["joint_default_pos"] = [float(default_pos[i]) for i in range(len(default_pos))]
    metadata["body_names"] = [name for name in robot.data.body_names]
    metadata["robot_num_bodies"] = len(robot.data.body_names)
    metadata["terrain_type"] = terrain_type
    metadata["cmd_sequence"] = args_cli.cmd_sequence
    metadata["contact_num_bodies"] = -1

    foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    foot_indices = []
    for fn in foot_names:
        try:
            foot_indices.append(list(robot.data.body_names).index(fn))
        except ValueError:
            foot_indices.append(-1)
    metadata["foot_indices"] = {fn: fi for fn, fi in zip(foot_names, foot_indices)}

    # 传感器中的足端索引（第一帧后填入）
    contact_foot_indices = [-1, -1, -1, -1]

    json_path = None
    if args_cli.record_physics:
        json_path = csv_file_path.replace(".csv", "_metadata.json")
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[INFO] 元信息已保存至: {json_path}")

        field_rows = build_field_descriptions(robot, foot_names)
        fields_path = csv_file_path.replace(".csv", "_fields.csv")
        with open(fields_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["字段名", "含义", "单位", "映射/备注"])
            for row in field_rows:
                writer.writerow(row)
        print(f"[INFO] 字段含义已保存至: {fields_path}")

    has_height_scanner = "height_scanner" in env.unwrapped.scene.keys()
    print(f"[INFO] 足端刚体索引: {foot_indices}")

    has_base_vel_cmd = hasattr(env.unwrapped, "command_manager")
    joint_limits_lower = metadata["joint_limits_lower"]
    joint_limits_upper = metadata["joint_limits_upper"]

    obs = env.get_observations()
    sim_step = 0
    video_step = 0
    record_step = 0
    header_written = False

    def safe_get(arr, idx=0, default=None):
        try:
            if arr is not None and isinstance(arr, torch.Tensor) and arr.numel() > 0:
                if arr.dim() >= 2 and arr.shape[0] > idx:
                    return arr[idx].detach().cpu().numpy().flatten().tolist()
                else:
                    return arr.detach().cpu().numpy().flatten().tolist()
        except Exception:
            pass
        return default if default is not None else []

    def reorder_by_body_names(data_flat, src_names, tgt_names):
        """将 src_names 顺序的扁平数据按 tgt_names 重排。每个刚体占 3 个值。"""
        if not data_flat or not src_names:
            return data_flat
        tgt_to_src = []
        for name in tgt_names:
            try:
                tgt_to_src.append(src_names.index(name))
            except ValueError:
                tgt_to_src.append(-1)
        result = []
        for si in tgt_to_src:
            if si >= 0 and si * 3 + 2 < len(data_flat):
                result.extend([data_flat[si*3], data_flat[si*3+1], data_flat[si*3+2]])
            else:
                result.extend([0.0, 0.0, 0.0])
        return result

    def reorder_scalar_by_body_names(data_scalar, src_names, tgt_names):
        """将 src_names 顺序的标量数据按 tgt_names 重排。每个刚体占 1 个值。"""
        if not data_scalar or not src_names:
            return data_scalar
        result = []
        for name in tgt_names:
            try:
                si = src_names.index(name)
                result.append(data_scalar[si] if si < len(data_scalar) else 0.0)
            except ValueError:
                result.append(0.0)
        return result

    try:
        while simulation_app.is_running():
            start_time = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                obs, rewards, dones, infos = env.step(actions)

                if args_cli.record_physics and (sim_step % args_cli.record_interval == 0):
                    if record_step == 0:
                        if contact_sensor is not None:
                            nf_first = safe_get(contact_sensor.data.net_forces_w)
                            metadata["contact_num_bodies"] = len(nf_first) // 3 if nf_first else 0
                            metadata["contact_body_names"] = list(contact_sensor.body_names)
                            metadata["contact_sensor_attrs"] = [a for a in dir(contact_sensor) if not a.startswith('_')]
                            metadata["contact_sensor_data_attrs"] = [a for a in dir(contact_sensor.data) if not a.startswith('_')]
                            if hasattr(contact_sensor, 'body_physx_view'):
                                metadata["body_physx_view_attrs"] = [a for a in dir(contact_sensor.body_physx_view) if not a.startswith('_')]
                            # 计算传感器中正确的足端索引
                            cb = list(contact_sensor.body_names)
                            metadata["verify_correct_foot_indices"] = {}
                            for fn in foot_names:
                                try:
                                    metadata["verify_correct_foot_indices"][fn] = cb.index(fn)
                                except ValueError:
                                    metadata["verify_correct_foot_indices"][fn] = -1
                            contact_foot_indices = [metadata["verify_correct_foot_indices"][fn] for fn in foot_names]
                            print(f"[INFO] contact_num_bodies = {metadata['contact_num_bodies']}")

                        jnames = list(robot.joint_names)
                        metadata["verify_joint_names_len"] = len(jnames)
                        metadata["verify_joint_pos_len"] = len(safe_get(robot.data.joint_pos))
                        metadata["verify_joint_vel_len"] = len(safe_get(robot.data.joint_vel))
                        metadata["verify_joint_tor_len"] = len(safe_get(robot.data.applied_torque))

                        bnames = list(robot.data.body_names)
                        bpos = safe_get(robot.data.body_pos_w) if hasattr(robot.data, 'body_pos_w') else []
                        bvel = safe_get(robot.data.body_lin_vel_w) if hasattr(robot.data, 'body_lin_vel_w') else []
                        metadata["verify_body_names_len"] = len(bnames)
                        metadata["verify_body_pos_bodies"] = len(bpos) // 3 if bpos else 0
                        metadata["verify_body_vel_bodies"] = len(bvel) // 3 if bvel else 0

                        metadata["verify_foot_names_by_index"] = {}
                        for fn, fi in zip(foot_names, foot_indices):
                            if 0 <= fi < len(bnames):
                                metadata["verify_foot_names_by_index"][fn] = bnames[fi]
                            else:
                                metadata["verify_foot_names_by_index"][fn] = f"INVALID({fi})"

                        if json_path:
                            with open(json_path, "w") as f:
                                json.dump(metadata, f, indent=2)

                    cmd = safe_get(env.unwrapped.command_manager.get_command("base_velocity")) if has_base_vel_cmd else []
                    pos = safe_get(robot.data.root_pos_w)
                    quat = safe_get(robot.data.root_quat_w)
                    lin = safe_get(robot.data.root_lin_vel_w)
                    ang = safe_get(robot.data.root_ang_vel_w)
                    act = actions[0].detach().cpu().numpy().tolist()
                    jpos = safe_get(robot.data.joint_pos)
                    jvel = safe_get(robot.data.joint_vel)
                    jtor = safe_get(robot.data.applied_torque)

                    net_forces_raw, contact_pos_raw = [], []
                    current_air_time_raw, last_air_time_raw, current_contact_time_raw = [], [], []
                    if contact_sensor is not None:
                        try:
                            net_forces_raw = safe_get(contact_sensor.data.net_forces_w)
                            contact_pos_raw = safe_get(contact_sensor.data.contact_pos_w)
                            current_air_time_raw = safe_get(contact_sensor.data.current_air_time)
                            last_air_time_raw = safe_get(contact_sensor.data.last_air_time)
                            current_contact_time_raw = safe_get(contact_sensor.data.current_contact_time)
                        except Exception:
                            pass

                    # 重排传感器数据到 body_names 顺序
                    bnames = list(robot.data.body_names)
                    if contact_sensor is not None and contact_foot_indices[0] >= 0:
                        cb_names = list(contact_sensor.body_names)
                        net_forces = reorder_by_body_names(net_forces_raw, cb_names, bnames)
                        contact_pos = reorder_by_body_names(contact_pos_raw, cb_names, bnames)
                        current_air_time = reorder_scalar_by_body_names(current_air_time_raw, cb_names, bnames)
                        last_air_time = reorder_scalar_by_body_names(last_air_time_raw, cb_names, bnames)
                        current_contact_time = reorder_scalar_by_body_names(current_contact_time_raw, cb_names, bnames)
                    else:
                        net_forces = net_forces_raw
                        contact_pos = contact_pos_raw
                        current_air_time = current_air_time_raw
                        last_air_time = last_air_time_raw
                        current_contact_time = current_contact_time_raw

                    root_com_pos = safe_get(robot.data.root_com_pos_w) if hasattr(robot.data, 'root_com_pos_w') else []
                    projected_gravity = safe_get(robot.data.projected_gravity_b) if hasattr(robot.data, 'projected_gravity_b') else []
                    heading = safe_get(robot.data.heading_w) if hasattr(robot.data, 'heading_w') else []

                    height_scan = []
                    if has_height_scanner:
                        try:
                            hs = env.unwrapped.scene["height_scanner"]
                            hits = hs.data.ray_hits_w[0]
                            height_scan = hits.detach().cpu().numpy().flatten().tolist() if hits.numel() > 0 else []
                        except Exception:
                            pass

                    body_pos_all = safe_get(robot.data.body_pos_w) if hasattr(robot.data, 'body_pos_w') else []
                    body_lin_vel_all = safe_get(robot.data.body_lin_vel_w) if hasattr(robot.data, 'body_lin_vel_w') else []
                    foot_pos, foot_vel, foot_height = [], [], []
                    num_bodies = len(body_pos_all) // 3
                    for fi in foot_indices:
                        if fi >= 0 and fi < num_bodies:
                            foot_pos.extend([body_pos_all[fi*3], body_pos_all[fi*3+1], body_pos_all[fi*3+2]])
                            foot_vel.extend([body_lin_vel_all[fi*3], body_lin_vel_all[fi*3+1], body_lin_vel_all[fi*3+2]])
                            foot_height.append(body_pos_all[fi*3+2] - pos[2] if len(pos) > 2 else 0.0)
                        else:
                            foot_pos.extend([0, 0, 0])
                            foot_vel.extend([0, 0, 0])
                            foot_height.append(0.0)

                    limit_violation = []
                    if len(jpos) > 0 and len(joint_limits_lower) > 0:
                        for i in range(min(len(jpos), len(joint_limits_lower))):
                            v = 1.0 if jpos[i] < joint_limits_lower[i] else (1.0 if jpos[i] > joint_limits_upper[i] else 0.0)
                            limit_violation.append(v)

                    jpw = []
                    if len(jvel) > 0 and len(jtor) > 0:
                        mlen = min(len(jvel), len(jtor))
                        jpw = [jtor[i] * jvel[i] for i in range(mlen)]

                    total_cf_mag = 0.0
                    if len(net_forces) >= 3:
                        nb = len(net_forces) // 3
                        for i in range(nb):
                            total_cf_mag += np.sqrt(net_forces[i*3]**2 + net_forces[i*3+1]**2 + net_forces[i*3+2]**2)

                    if not header_written:
                        csv_file = open(csv_file_path, "w", newline="")
                        csv_writer = csv.writer(csv_file)
                        h = ["time", "record_step"]
                        h += [f"cmd_{i}" for i in range(len(cmd))]
                        h += ["pos_z", "quat_w", "quat_x", "quat_y", "quat_z"]
                        h += ["lin_x", "lin_y", "lin_z", "ang_x", "ang_y", "ang_z"]
                        h += [f"action_{i}" for i in range(len(act))]
                        h += [f"joint_pos_{i}" for i in range(len(jpos))]
                        h += [f"joint_vel_{i}" for i in range(len(jvel))]
                        h += [f"applied_torque_{i}" for i in range(len(jtor))]
                        h += [f"joint_power_{i}" for i in range(len(jpw))]
                        h += [f"joint_limit_violation_{i}" for i in range(len(limit_violation))]
                        h += [f"net_force_{i}" for i in range(len(net_forces))]
                        h += [f"contact_pos_{i}" for i in range(len(contact_pos))]
                        h += [f"current_air_time_{i}" for i in range(len(current_air_time))]
                        h += [f"last_air_time_{i}" for i in range(len(last_air_time))]
                        h += [f"current_contact_time_{i}" for i in range(len(current_contact_time))]
                        h += [f"root_com_pos_{i}" for i in range(len(root_com_pos))]
                        h += [f"projected_gravity_{i}" for i in range(len(projected_gravity))]
                        h += ["heading_w"] if heading else []
                        h += [f"height_scan_{i}" for i in range(len(height_scan))]
                        for fn in foot_names:
                            h += [f"foot_{fn}_pos_x", f"foot_{fn}_pos_y", f"foot_{fn}_pos_z"]
                        for fn in foot_names:
                            h += [f"foot_{fn}_vel_x", f"foot_{fn}_vel_y", f"foot_{fn}_vel_z"]
                        h += [f"foot_height_{fn}" for fn in foot_names]
                        h += ["total_contact_force_magnitude"]
                        csv_writer.writerow(h)
                        csv_file.flush()
                        header_written = True
                        print(f"[INFO] 表头已写入，共 {len(h)} 列")

                    row = [sim_step * dt, record_step]
                    row += cmd + [pos[2]] + quat + lin + ang + act + jpos + jvel + jtor + jpw
                    row += limit_violation
                    row += net_forces + contact_pos + current_air_time + last_air_time + current_contact_time
                    row += root_com_pos + projected_gravity
                    if heading:
                        row += heading
                    row += height_scan
                    row += foot_pos + foot_vel + foot_height
                    row += [total_cf_mag]
                    csv_writer.writerow(row)
                    csv_file.flush()
                    track_count += 1
                    record_step += 1

                policy_nn.reset(dones)

            sim_step += 1
            if args_cli.video:
                video_step += 1
                if video_step == args_cli.video_length:
                    break

            if total_duration is not None and sim_step * dt >= total_duration:
                print(f"[INFO] 命令序列执行完毕 ({total_duration:.1f}s), 停止仿真")
                break

            if args_cli.keyboard:
                camera_follow(env)

            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        if csv_file is not None:
            csv_file.close()
            if csv_file_path and track_count > 0:
                print(f"[INFO] 文件已关闭: {csv_file_path}")
                print(f"[INFO] 共写入 {track_count} 步数据 ({os.path.getsize(csv_file_path)/1024:.1f} KB)")

    env.close()


if __name__ == "__main__":
    _register_env(args_cli.task)
    main()
    simulation_app.close()