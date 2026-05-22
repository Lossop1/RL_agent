# training 禁止规则

## 禁止操作
- 禁止 parse_training_log（不分析日志）
- 禁止 deep_dig（不挖掘快照）
- 禁止 tuner（不改奖励）
- 禁止在启动训练前自己探查远程目录结构（路径由 knowledge/robot_lab.md 和 state/env_state.json 提供）
- 禁止使用 --resume True（应使用 --resume 纯 flag）

## switch_context 限制
- 禁止 switch_context 到未在 skill 中明确定义的 context_name
- 允许的 context_name: env_setup
- 只有在步骤 0.1 校验不通过时才允许 switch_context


