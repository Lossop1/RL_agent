# RobotLab 项目结构参考

## 训练脚本路径

训练脚本位于 `robot_lab/scripts/reinforcement_learning/` 目录下，按框架分目录：

```
robot_lab/scripts/reinforcement_learning/
├── cusrl/
│   ├── play.py
│   └── train.py
├── rl_utils.py
├── rsl_rl/
│   ├── cli_args.py
│   ├── play.py
│   ├── play_cs.py
│   └── train.py
└── skrl/
    ├── play.py
    └── train.py
```

## 机器人配置目录

每个机器人对应一个目录，位于 `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/` 下。每个目录包含：

- `flat_env_cfg.py` — 平坦地形环境配置（含奖励权重）
- `rough_env_cfg.py` — 粗糙地形环境配置（含奖励权重）
- `agents/` — RL 算法参数配置

目录结构：

```
source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/
├── agibot_d1/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── cusrl_ppo_cfg.py
│   │   └── rsl_rl_ppo_cfg.py
│   ├── flat_env_cfg.py
│   └── rough_env_cfg.py
├── anymal_d/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── cusrl_distillation_cfg.py
│   │   ├── cusrl_ppo_cfg.py
│   │   ├── rsl_rl_distillation_cfg.py
│   │   └── rsl_rl_ppo_cfg.py
│   ├── flat_env_cfg.py
│   └── rough_env_cfg.py
├── deeprobotics_lite3/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── cusrl_ppo_cfg.py
│   │   └── rsl_rl_ppo_cfg.py
│   ├── flat_env_cfg.py
│   └── rough_env_cfg.py
├── magiclab_magicdog/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── cusrl_ppo_cfg.py
│   │   └── rsl_rl_ppo_cfg.py
│   ├── flat_env_cfg.py
│   └── rough_env_cfg.py
├── unitree_a1/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── cusrl_ppo_cfg.py
│   │   └── rsl_rl_ppo_cfg.py
│   ├── flat_env_cfg.py
│   └── rough_env_cfg.py
├── unitree_b2/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── cusrl_ppo_cfg.py
│   │   └── rsl_rl_ppo_cfg.py
│   ├── flat_env_cfg.py
│   └── rough_env_cfg.py
├── unitree_go2/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── cusrl_ppo_cfg.py
│   │   └── rsl_rl_ppo_cfg.py
│   ├── flat_env_cfg.py
│   └── rough_env_cfg.py
└── zsibot_zsl1/
    ├── __init__.py
    ├── agents/
    │   ├── __init__.py
    │   └── rsl_rl_ppo_cfg.py
    ├── flat_env_cfg.py
    └── rough_env_cfg.py
```

## 日志目录

训练日志和模型保存在 `~/robot_lab/logs/` 下，按框架分：

```
~/robot_lab/logs/
├── cusrl/
└── rsl_rl/
```

每个训练运行会创建一个时间戳子目录，包含 TensorBoard 事件文件、模型权重和参数配置。

## 训练命令参数

训练脚本位于 `scripts/reinforcement_learning/rsl_rl/train.py`：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task=<TASK_NAME> --headless
```

常用参数：

| 参数 | 说明 | 示例 |
|------|------|------|
| `--task` | gym 注册的任务名（必填） | `RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0` |
| `--headless` | 无 GUI 模式 | |
| `--num_envs` | 并行环境数 | `--num_envs 5000` |
| `--max_iterations` | 最大迭代轮数 | `--max_iterations 10000` |
| `--resume` | 从检查点恢复（纯 flag，不需要传值） | `--resume` |
| `--load_run` | 要恢复的运行目录名（时间戳） | `--load_run 2026-05-12_13-33-58` |
| `--checkpoint` | 指定检查点文件路径（可选，不指定则自动找最新） | `--checkpoint /PATH/TO/model.pt` |
| `--agent` | 指定 agent 配置入口点 | `--agent=rsl_rl_ppo_cfg_entry_point` |
| `--run_name` | 自定义运行名称 | `--run_name=my_experiment` |
| `--video` | 录制视频 | `--video --video_length 200` |
| `--keyboard` | 键盘控制（play 模式） | `--keyboard` |
| `--distributed` | 多 GPU 训练 | `--distributed` |

> `--resume` 是 argparse `store_true` 标志，直接写 `--resume` 即可，不要写 `--resume True` 或 `--resume=True`。

## 命名规则

### experiment_name（日志目录名）

格式：`{model}_{terrain}`

- model 映射：go2 → unitree_go2, h1 → unitree_h1, a1 → unitree_a1, b2 → unitree_b2
- terrain: rough 或 flat
- 例：go2 + rough → `unitree_go2_rough`

### task_name（--task 参数值）

格式：`RobotLab-Isaac-Velocity-{Terrain}-{Manufacturer}-{Model}-v0`

例：`RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0`

### 日志路径

```
~/robot_lab/logs/rsl_rl/{experiment_name}/{timestamp}/
├── model_X.pt          # 检查点
├── train.log           # 训练日志
├── params/             # 训练参数快照
│   ├── env.yaml
│   └── agent.yaml
└── events.out.tfevents.*  # TensorBoard 事件
```

## 任务注册机制

每个机器人的 `__init__.py` 通过 gym.register() 注册任务：

```python
gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg",
    },
)
```

- `env_cfg_entry_point` 指向奖励配置类
- `rsl_rl_cfg_entry_point` 指向 PPO 算法参数类

## 可用任务列表

以下是所有可用的训练任务（`--task` 参数值）：

| 类别 | 机器人 | 任务名 |
|------|--------|--------|
| 四足 | Anymal D | RobotLab-Isaac-Velocity-Rough-Anymal-D-v0 |
| 四足 | Unitree Go2 | RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0 |
| 四足 | Unitree Go2 | RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 |
| 四足 | Unitree B2 | RobotLab-Isaac-Velocity-Rough-Unitree-B2-v0 |
| 四足 | Unitree B2 | RobotLab-Isaac-Velocity-Flat-Unitree-B2-v0 |
| 四足 | Unitree A1 | RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0 |
| 四足 | Unitree A1 | RobotLab-Isaac-Velocity-Flat-Unitree-A1-v0 |
| 四足 | Deeprobotics Lite3 | RobotLab-Isaac-Velocity-Rough-Deeprobotics-Lite3-v0 |
| 四足 | Deeprobotics Lite3 | RobotLab-Isaac-Velocity-Flat-Deeprobotics-Lite3-v0 |
| 四足 | Zsibot ZSL1 | RobotLab-Isaac-Velocity-Rough-Zsibot-ZSL1-v0 |
| 四足 | Zsibot ZSL1 | RobotLab-Isaac-Velocity-Flat-Zsibot-ZSL1-v0 |
| 四足 | Magiclab MagicDog | RobotLab-Isaac-Velocity-Rough-MagicLab-Dog-v0 |
| 四足 | Magiclab MagicDog | RobotLab-Isaac-Velocity-Flat-MagicLab-Dog-v0 |
| 四足 | Agibot D1 | RobotLab-Isaac-Velocity-Rough-Agibot-D1-v0 |
| 四足 | Agibot D1 | RobotLab-Isaac-Velocity-Flat-Agibot-D1-v0 |
| 轮式 | Unitree Go2W | RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0 |
| 轮式 | Unitree Go2W | RobotLab-Isaac-Velocity-Flat-Unitree-Go2W-v0 |
| 轮式 | Unitree B2W | RobotLab-Isaac-Velocity-Rough-Unitree-B2W-v0 |
| 轮式 | Unitree B2W | RobotLab-Isaac-Velocity-Flat-Unitree-B2W-v0 |
| 轮式 | Deeprobotics M20 | RobotLab-Isaac-Velocity-Rough-Deeprobotics-M20-v0 |
| 轮式 | Deeprobotics M20 | RobotLab-Isaac-Velocity-Flat-Deeprobotics-M20-v0 |
| 轮式 | DDTRobot Tita | RobotLab-Isaac-Velocity-Rough-DDTRobot-Tita-v0 |
| 轮式 | DDTRobot Tita | RobotLab-Isaac-Velocity-Flat-DDTRobot-Tita-v0 |
| 轮式 | Zsibot ZSL1W | RobotLab-Isaac-Velocity-Rough-Zsibot-ZSL1W-v0 |
| 轮式 | Zsibot ZSL1W | RobotLab-Isaac-Velocity-Flat-Zsibot-ZSL1W-v0 |
| 轮式 | Magiclab MagicDog-W | RobotLab-Isaac-Velocity-Rough-MagicLab-Dog-W-v0 |
| 轮式 | Magiclab MagicDog-W | RobotLab-Isaac-Velocity-Flat-MagicLab-Dog-W-v0 |
| 人形 | Unitree G1 | RobotLab-Isaac-Velocity-Rough-Unitree-G1-v0 |
| 人形 | Unitree G1 | RobotLab-Isaac-Velocity-Flat-Unitree-G1-v0 |
| 人形 | Unitree H1 | RobotLab-Isaac-Velocity-Rough-Unitree-H1-v0 |
| 人形 | Unitree H1 | RobotLab-Isaac-Velocity-Flat-Unitree-H1-v0 |
| 人形 | FFTAI GR1T1 | RobotLab-Isaac-Velocity-Rough-FFTAI-GR1T1-v0 |
| 人形 | FFTAI GR1T1 | RobotLab-Isaac-Velocity-Flat-FFTAI-GR1T1-v0 |
| 人形 | FFTAI GR1T2 | RobotLab-Isaac-Velocity-Rough-FFTAI-GR1T2-v0 |
| 人形 | FFTAI GR1T2 | RobotLab-Isaac-Velocity-Flat-FFTAI-GR1T2-v0 |
| 人形 | Booster T1 | RobotLab-Isaac-Velocity-Rough-Booster-T1-v0 |
| 人形 | Booster T1 | RobotLab-Isaac-Velocity-Flat-Booster-T1-v0 |
| 人形 | RobotEra Xbot | RobotLab-Isaac-Velocity-Rough-RobotEra-Xbot-v0 |
| 人形 | RobotEra Xbot | RobotLab-Isaac-Velocity-Flat-RobotEra-Xbot-v0 |
| 人形 | Openloong Loong | RobotLab-Isaac-Velocity-Rough-Openloong-Loong-v0 |
| 人形 | Openloong Loong | RobotLab-Isaac-Velocity-Flat-Openloong-Loong-v0 |
| 人形 | RoboParty ATOM01 | RobotLab-Isaac-Velocity-Rough-RoboParty-ATOM01-v0 |
| 人形 | RoboParty ATOM01 | RobotLab-Isaac-Velocity-Flat-RoboParty-ATOM01-v0 |
| 人形 | Magiclab MagicBot-Gen1 | RobotLab-Isaac-Velocity-Rough-MagicLab-Bot-Gen1-v0 |
| 人形 | Magiclab MagicBot-Gen1 | RobotLab-Isaac-Velocity-Flat-MagicLab-Bot-Gen1-v0 |
| 人形 | Magiclab MagicBot-Z1 | RobotLab-Isaac-Velocity-Rough-MagicLab-Bot-Z1-v0 |
| 人形 | Magiclab MagicBot-Z1 | RobotLab-Isaac-Velocity-Flat-MagicLab-Bot-Z1-v0 |
| 人形 | Unitree G1 (BeyondMimic) | RobotLab-Isaac-BeyondMimic-Flat-Unitree-G1-v0 |
| 人形 | Unitree G1 (AMP Dance) | RobotLab-Isaac-G1-AMP-Dance-Direct-v0 |
| 四足 | Unitree A1 (HandStand) | RobotLab-Isaac-Velocity-Flat-HandStand-Unitree-A1-v0 |


