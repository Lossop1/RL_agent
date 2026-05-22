import json

class StuckDetector:
    """
    Detects four pathological agent loop patterns by hashing tool call history.

    Scenarios:
      S1 — same tool + same args + same result, 4x consecutive (spinning)
      S2 — same tool + same args, 3x consecutive, all errors (retry loop)
      S3 — only 'think' tool calls, 5x consecutive (paralysis)
      S4 — alternating (A,B,A,B,A,B) pattern with identical results (oscillation)
    """

    WINDOW = 20

    _MESSAGES = {
        "S1": (
            "你已对相同目标执行了完全相同的操作 4 次，得到相同结果。"
            "请改变方法或先诊断根本原因，再继续操作。"
        ),
        "S2": (
            "此操作已连续失败 3 次（参数相同）。"
            "停止重试。在再次尝试前，先用 bash 或 think 诊断失败原因。"
        ),
        "S3": (
            "你已连续推理 5 步而未执行任何操作。"
            "现在立即执行一个具体的 bash 或 file 操作。"
        ),
        "S4": (
            "你在两个操作之间来回交替，且每次结果相同，没有任何进展。"
            "这个组合不起作用。尝试完全不同的策略。"
        ),
    }

    def __init__(self):
        # Each entry: (tool_name, args_hash, result_hash, is_error)
        self.history: list[tuple[str, int, int, bool]] = []

    def record(self, tool_name: str, args: dict, result: dict) -> None:
        args_h   = hash(json.dumps(args,   sort_keys=True, default=str))
        result_h = hash(json.dumps(result, sort_keys=True, default=str))
        is_error = result.get("error") is not None
        self.history.append((tool_name, args_h, result_h, is_error))
        if len(self.history) > self.WINDOW:
            self.history = self.history[-self.WINDOW:]

    def check(self) -> tuple[bool, str]:
        """Return (is_stuck, scenario_id). scenario_id is '' if not stuck.

        Check order matters:
          S2 before S1  — error loops are more actionable than spin loops.
          S3 before S1  — 'think' results are deterministically identical by
                          design, so S1 must exclude 'think' to avoid masking S3.
        """
        h = self.history

        # S2: last 3 same (name, args), all errors — check before S1
        if len(h) >= 3:
            last3 = h[-3:]
            if (all(e[0] == last3[0][0] and e[1] == last3[0][1] for e in last3)
                    and all(e[3] for e in last3)):
                return True, "S2"

        # S3: last 5 all 'think' — check before S1 so 'think' loops aren't masked
        if len(h) >= 5 and all(e[0] == "think" for e in h[-5:]):
            return True, "S3"

        # S1: last 4 identical (name, args, result) — exclude 'think' (deterministic by design)
        if len(h) >= 4:
            last4 = h[-4:]
            if (last4[0][0] != "think"
                    and all(e[0] == last4[0][0] and e[1] == last4[0][1] and e[2] == last4[0][2]
                            for e in last4)):
                return True, "S1"

        # S4: last 6 form (A,B,A,B,A,B) with matching result hashes per slot
        # Identity = (tool_name, args_hash) so two bash calls with diff args count as A≠B
        if len(h) >= 6:
            last6 = h[-6:]
            idents = [(e[0], e[1]) for e in last6]
            res    = [e[2] for e in last6]
            key_a, key_b = idents[0], idents[1]
            if (key_a != key_b
                    and idents == [key_a, key_b, key_a, key_b, key_a, key_b]
                    and res[0::2] == [res[0]] * 3
                    and res[1::2] == [res[1]] * 3):
                return True, "S4"

        return False, ""

    def intervention_message(self, scenario_id: str) -> dict:
        """Return a user-role message to inject before the next LLM call."""
        text = self._MESSAGES.get(scenario_id, "检测到异常循环，请重新评估当前策略。")
        return {"role": "user", "content": f"[系统检测] {text}"}

    def reset(self) -> None:
        self.history.clear()
