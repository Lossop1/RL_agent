# meta 禁止规则

## 禁止操作
- 禁止 ssh_exec（任何远程操作）
- 禁止 file_read（任何文件读取）
- 禁止 think（不做任何思考）
- 禁止 ask_followup_question（不做任何询问）
- 禁止在 switch_context 之前执行任何其他操作

## switch_context 限制
- 禁止 switch_context 到未在 skill 中明确定义的 context_name
- 允许的 context_name: rectify, env_setup


