# env_setup 禁止规则

## 禁止操作
- 禁止 switch_context（env_setup 不启动其他 agent）
- 禁止推断 conda_env、platform、project——这些由调用者传入，env_setup 只验证
- 禁止不读 state/env_state.json 就执行完整检查（先查缓存）

## pop_context 限制
- 返回时必须包含 fingerprint、conda_env、conda_sh_path、work_dir、platform、project
- status 只能是 ok 或 error
