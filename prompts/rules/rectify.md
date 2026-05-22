# rectify 禁止规则

## 禁止操作
- 禁止 ssh_exec（任何远程操作都不行）
- 禁止 cmd（任何本地命令都不行）
- 禁止 tail_log（日志显示是 training 的职责）
- 禁止 file_write（不写任何文件）
- 禁止在步骤 1 中自己探查远程环境（应直接 switch_context 到 training）
- 禁止在步骤 3 和步骤 4 之间插入其他操作（parse_log 后必须立即 deep_dig）
- 禁止在步骤 5 中调用 ssh_exec 或读远程文件
- 禁止在步骤 5 中仅凭一两个指标就下结论。必须列出 parse_training_log 输出的所有关键指标，逐项分析趋势

## switch_context 限制
- 禁止 switch_context 到未在 skill 中明确定义的 context_name
- 允许的 context_name: training
- 只有在步骤 1 中才允许 switch_context

## 分析间隔


- 步骤 3-5 的完整分析（parse_log + deep_dig + 诊断）每 200-500 轮执行一次
- 步骤 2.2 的进度查看不在此限，可频繁执行
- 不允许计算超过 500 轮的间隔


