# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

RL-Dog Training Agent — 基于 LLM 的自主 Agent，通过 SSH 连接远程 GPU 服务器，自动监控四足机器人 RL 训练、分析奖励趋势、诊断行为品质差距、调整奖励权重、重启训练。目标是交付"自然、稳定、省力、听话"的运动策略。

## 运行

```bash
# 安装依赖
pip install openai paramiko numpy scipy scikit-learn

# 交互模式（进入 CLI 循环）
python main.py

# 非交互模式（直接指定任务）
python main.py "检查 Unitree Go2 的训练进度并分析奖励趋势"

# 冒烟测试
python validate.py
```

## 配置

启动前需要两个配置文件（不会被提交）：

- `config/llm.json` — `api_key`, `base_url`, `model`, `max_turns` (默认 300)
- `config/ssh.json` — `ssh_host`, `ssh_port`, `ssh_user`, `ssh_pass`, 可选 `shell_init`, `work_dir`

日志格式配置在 `config/log_formats/` 下。主题在 `config/theme.json`。

## 多 Agent 上下文系统

这是最核心的架构概念。系统有 4 个 sub-agent，各有独立的 system prompt、工具白名单和消息历史：

```
meta → rectify → training
            └──→ env_setup (由 training 调用)
```

- **meta** (`prompts/skills/meta.md`): 纯路由器。收到用户消息后立即 `switch_context` 到 `rectify`，不思考、不探查。
- **rectify** (`prompts/skills/rectify.md`): 核心调参逻辑。加载知识库→启动训练→监控→parse_log + deep_dig→行为诊断→根因推断→tuner 修改权重→restart 训练。最多 20 轮。
- **training** (`prompts/skills/training.md`): 训练执行器。5 个分支：附着已有(A)、修复日志管道(B)、检查点恢复(C)、冷启动(D)、调参后重启(E)。只负责启动和日志显示，不做分析。
- **env_setup** (`prompts/skills/env_setup.md`): 远程环境校验。SSH 连通性、conda 环境、工作目录、PyTorch 验证。结果缓存到 `state/env_state.json`(按 SSH 指纹索引)。

上下文通过 `switch_context` / `pop_context` 切换，`main.py` 维护 `context_stack` 和 `contexts` 字典，每轮结束后自动保存到 `sessions/contexts/`。

每个 agent 的 `allowed-tools` 在 skill 文件头部声明。`tools/__init__.py` 自动发现和注册所有工具。

Agent 的禁止规则在 `prompts/rules/{agent_name}.md` 中定义（由 `context_switch.py` 在创建 context 时加载合并到 system prompt）。

## 工具系统

工具在 `tools/` 下自动发现：模块名不以 `_` 开头、有 `TOOL` 字典和 `execute` 函数的即被注册。`tools/__init__.py` 的 `REGISTRY` 是全局工具注册表，`dispatch()` 是统一执行入口。

`main.py` 中的 `CONTEXT_TOOLS` 字典限制了每个 agent 可用的工具集——switch_context 时重新过滤。

工具定义使用简化类型系统（`str/int/float/bool/object`），`get_openai_tools()` 转换为 OpenAI function-calling 格式。

## 上下文压缩 (`context.py`)

三级策略，全部原地修改 `messages` 列表（保持引用一致）：

1. **MicroCompact** — 超过 `KEEP_RECENT_TOOLS=80` 条工具消息时，旧消息内容替换为 `[old result cleared]`
2. **AutoCompact** — 消息正文超过 `MAX_CONTEXT_CHARS=600000` 时，调 LLM 生成结构化摘要（保留 `[BEHAVIOR]`/`[GAP]`/`[ROOT_CAUSE]`/`[ACTION]` 标记段原文）
3. **EmergencySnip** — 上下文过大无法调 API 时，保留 system + 最近摘要 + 最后 4 条消息

带熔断器：连续 `MAX_COMPACT_RETRIES=3` 次失败后跳过压缩。

## 卡死检测 (`stuck.py`)

`StuckDetector` 维护最近 20 个工具调用的哈希历史，检测 4 种模式：

- **S1**: 相同工具+相同参数+相同结果 4 次 → 死循环
- **S2**: 相同工具+相同参数 3 次全部 error → 重试循环（优先级高于 S1）
- **S3**: 连续 5 次只调 `think` → 分析瘫痪（优先级高于 S1）
- **S4**: (A,B,A,B,A,B) 交替且结果相同 → 振荡

检测到后注入干预消息打断循环。

## 安全边界

- `tuner.py` 只能修改 `/source/robot_lab/robot_lab/tasks/` 下的 `self.rewards.*.weight` 字段
- `main.py` 的 `_needs_confirmation()` 对危险 SSH 命令（`sed -i`, `rm`, `mv`, `>`, `pip install` 等）和写非 sessions 目录的文件要求用户确认
- `bash.py` 拦截 `rm -rf /`、`mkfs`、`dd if=` 等危险命令
- rectify 禁止自己 kill 进程或启动训练脚本——重启必须通过 `switch_context` 调遣 training

## 训练日志解析 (`parse_log.py`)

使用 numpy/scipy/sklearn 的完整分析流水线。核心指标：
- 趋势（线性回归斜率）、单调性、Mann-Kendall 显著性、变异系数
- Pearson/Spearman/Kendall 相关性 + 偏相关 + 领先-滞后
- 奖励聚类分解（smoothness/energy/gait/constraint/stability/tracking）
- 窗口大小自适应（根据 total_iterations 缩放）

## deep_dig 深度分析

- PELT 变点检测（均值结构性变化）
- 交叉相关 + 格兰杰因果（事件关联）
- 孤立森林异常检测（n_estimators=100, contamination=0.05）
- Holt-Winters 趋势预测（季节性周期=10）
- Mann-Whitney U 调参前后对比

## 知识库

- `knowledge/quadruped_diagnosis.md` — 指标→行为→根因→调参的完整知识图谱（冲突对、协同对、层级依赖、8 种行为问题模式、作弊检测）
- `knowledge/robot_lab.md` — 40+ 可用任务、命名规则、训练命令参数

## 重要约定

- prompt/skill 文件头部用 `allowed-tools:` 声明工具白名单，不能使用未列出的工具
- `context_switch.py` 加载 skill 时自动合并 rules 目录下的通用规则和 agent 专用规则
- `pop_context` 返回时，子 agent 的消息历史保留在 context 字典中，并在父 agent 中追加一行 `[来自 {child}] 任务完成: ...` 摘要
- 禁止 `--resume True`，必须用纯 flag `--resume`



## 行为准则 (来自 Karpathy 指南)

### 1. 编码前思考
- **明确说出你的假设**。不确定就问，别猜。
- **有歧义时，列出所有可能的解释和利弊**，别自己默默选一个。
- **如果我的要求有更简单或更好的方法，直接指出来。**
- **感到困惑时立刻停下**，指出哪里不清楚，等我澄清。

### 2. 简洁优先 (YAGNI)
- 用最精简的代码完成我要求的事。
- **不要** 添加任何我未要求的功能、选项或"灵活性"。
- **不要** 为一次性代码创建抽象层。
- **不要** 为几乎不可能发生的情况编写复杂的错误处理。
- 如果你发现复杂代码可以大幅精简，提出来。

### 3. 精准修改
- **只改动与我请求直接相关的代码。**
- **不要** "顺便"改进、重构或重新格式化无关的代码。
- **不要** 修改或删除你不理解的现有注释。
- **严格遵守** 项目现有的代码风格，即使你不喜欢。
- 如果你的改动让某些旧代码变成死代码，提出来征求我的意见，不要自行删除。

### 4. 目标驱动执行
- 不要只接受指令，要和我一起定义**可验证的成功标准**。
- 始终优先考虑**测试驱动**的方法：先写测试，再写实现。
- 开始复杂任务前，先简要说明你的计划，并标明每一步的验证方式。
