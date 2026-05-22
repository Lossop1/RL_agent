# RL-Dog Training Agent

基于 LLM 的自主 Agent，用于四足机器人强化学习训练的奖励调参。

通过 SSH 连接到远程训练机，读取训练日志，分析奖励趋势，自动调整奖励权重，直到策略达到预期的行为品质。

## 架构

```
main.py
 ├── config_loader.py        加载 LLM/SSH/prompt 配置
 ├── context.py              三级上下文压缩（MicroCompact / AutoCompact / EmergencySnip）
 ├── runtime_state.py        运行时配置状态共享
 ├── stuck.py                检测 Agent 卡死模式（S1-S4）
 ├── tools/__init__.py       工具自动注册与分发
 │    ├── bash.py            远程 shell 执行
 │    ├── cmd.py             本地 cmd 执行
 │    ├── ssh_exec.py        SSH 命令执行
 │    ├── file_read.py       文件读取
 │    ├── file_write.py      文件写入
 │    ├── file_edit.py       文件编辑
 │    ├── parse_log.py       训练日志解析（趋势、波动率、惩罚结构、偏相关）
 │    ├── deep_dig.py        深度分析（变化点、异常检测、趋势预测）
 │    ├── tuner.py           奖励权重调整
 │    ├── tail_log.py        实时日志跟踪
 │    ├── think.py           思考记录
 │    ├── context_switch.py  上下文切换
 │    ├── experience_query.py 经验查询
 │    ├── experience_write.py 经验写入
 │    ├── training_status.py  训练状态查询
 │    ├── task_complete.py    任务完成
 │    ├── grep.py             文件搜索
 │    └── glob_util.py        文件路径匹配
 ├── config/
 │    ├── llm.json            LLM 模型配置
 │    ├── ssh.json            SSH 连接配置
 │    ├── theme.json          CLI 主题
 │    └── thresholds.json     指标收敛/停滞判定阈值
 ├── prompts/
 │    ├── skills/
 │    │    ├── meta.md        系统 prompt（角色定义）
 │    │    └── style.md       输出风格
 │    └── preferences/
 │         ├── remote_system.md  远程机与系统环境偏好
 │         └── training.md       训练参数偏好
 ├── knowledge/
 │    ├── quadruped_diagnosis.md  四足机器人奖励调参知识库
 │    └── robot_lab.md            RobotLab 项目结构参考
 └── sessions/                会话持久化
      ├── contexts/           上下文快照
      └── trace_*.log         会话日志
```

## 快速开始

### 依赖

```bash
pip install openai
```

其余全用 Python 标准库。

### 配置

**config/llm.json**

```json
{
  "api_key": "sk-xxx",
  "base_url": "https://api.openai.com/v1",
  "model": "gpt-4o",
  "max_turns": 300
}
```

**config/ssh.json**

```json
{
  "ssh_host": "远程IP",
  "ssh_port": "22",
  "ssh_user": "用户名",
  "ssh_pass": "密码"
}
```

Agent 通过 SSH 连接到远程训练机执行所有操作（启动训练、读取日志、修改奖励权重）。本地不需要安装 Isaac Lab 或 RobotLab(但要求远程机的环境已经配好)。当前仅支持一台远程机，以及启动并监控robot_lab的训练。

### 启动

```bash
# 交互模式
python main.py

# 直接指定任务（非交互）
python main.py "检查 Unitree Go2 的训练进度并分析奖励趋势"
```

## 会话管理

Agent 每轮结束后自动保存上下文到 `sessions/contexts/`。

启动时显示历史会话列表：

```
  1. S2026-05-19_10-30-00
       2026-05-19 10:30:00 · turn 42 · 检查 Unitree Go2 训练进度...
```

- 输入编号恢复会话
- 输入 `d<编号>` 删除会话
- 直接输入文本开始新会话

会话日志保存在 `sessions/trace_*.log`。

## 工作流程

1. Agent 接收任务（如"分析训练进度并调参"）
2. 通过 SSH 连接远程训练机
3. 读取训练日志，解析奖励趋势、惩罚结构、偏相关矩阵
4. 根据知识库中的调参原则判断根因
5. 修改奖励配置文件中的权重
6. 等待冷却期后验证效果
7. 重复直到任务完成
   ...
调参效果不尽如人意，交付业务也还未完善

## 工具列表

| 工具 | 功能 |
|------|------|
| bash | 远程 shell 执行 |
| cmd | 本地 cmd 执行 |
| ssh_exec | SSH 命令执行 |
| file_read | 读取文件 |
| file_write | 写入文件 |
| file_edit | 编辑文件 |
| parse_training_log | 解析训练日志（趋势、波动率、惩罚结构、偏相关） |
| deep_dig | 深度分析（变化点、异常检测、趋势预测） |
| tuner | 调整奖励权重 |
| tail_log | 实时跟踪训练日志 |
| think | 记录思考过程 |
| switch_context | 切换上下文 |
| pop_context | 返回上级上下文 |
| experience_query | 查询历史经验 |
| experience_write | 写入历史经验 |
| training_status | 查询训练状态 |
| task_complete | 标记任务完成 |
| grep | 文件内容搜索 |
| glob_util | 文件路径匹配 |

## 知识库

- `knowledge/quadruped_diagnosis.md` — 四足机器人奖励调参知识库，包含指标解读、冲突对、层级依赖、作弊模式、调参原则
- `knowledge/robot_lab.md` — RobotLab 项目结构参考，包含训练脚本路径、机器人配置目录、训练命令参数、任务注册机制
