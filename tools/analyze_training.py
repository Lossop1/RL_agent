"""analyze_training.py — parse_log + deep_dig 合并工具

一步完成日志解析、趋势分析、深度挖掘。
"""
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from tools.parse_log import (
    _fetch_remote_log, load_format_config, tail_log,
    extract_total_iterations, parse_log_to_snapshots, LogBuffer,
    compute_adaptive_windows, check_data_availability, compute_all,
    TAIL_LINES, DEFAULT_TOTAL_ITER,
)
from tools.deep_dig import (
    compute_all_change_points, compute_all_event_links,
    compute_anomalies, compute_trend_forecast,
    compute_before_after, _find_previous_snapshot, _load_snapshot, _clean_for_json,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


TOOL = {
    "name": "analyze_training",
    "description": (
        "一步完成训练日志解析和深度分析。"
        "包含：趋势计算、相关性分析、变点检测、事件关联、异常检测、趋势预测。"
        "代替 parse_training_log + deep_dig 分步调用。"
    ),
    "parameters": {
        "log_path": {
            "type": "str", "required": True,
            "desc": "训练日志文件路径。本地路径或远程绝对路径。"
        },
        "total_iterations": {
            "type": "int", "required": False,
            "desc": "训练总轮数。不传则从日志中自动提取。"
        },
        "log_format": {
            "type": "str", "required": False,
            "desc": "日志格式配置名称。默认 'isaac_rl'。"
        },
        "remote": {
            "type": "bool", "required": False,
            "desc": "日志在远程服务器上。true 时自动从 config/ssh.json 读取凭据。"
        },
        "data_dir": {
            "type": "str", "required": False,
            "desc": "快照目录。传此参数时会做调参前后效果对比。"
        },
        "deep": {
            "type": "bool", "required": False,
            "desc": "是否返回深度分析数据（变点、事件关联、异常检测等）。默认 false。深度数据始终写入快照文件。"
        },
    }
}


def execute(log_path: str, total_iterations: int = None,
            log_format: str = "isaac_rl", remote: bool = False,
            data_dir: str = None, deep: bool = False,
            _control: dict = None) -> dict:
    """执行完整训练分析（parse_log + deep_dig）。"""
    try:
        # ── 1. 获取日志 ──
        if remote:
            log_path = _fetch_remote_log(log_path, TAIL_LINES)

        format_config = load_format_config(log_format)
        lines = tail_log(log_path, TAIL_LINES)

        if total_iterations is None:
            total_iterations = extract_total_iterations(lines, format_config)
        if total_iterations is None:
            total_iterations = DEFAULT_TOTAL_ITER

        windows = compute_adaptive_windows(total_iterations)

        # ── 2. 创建/更新缓冲区 ──
        if remote:
            safe_name = log_path.replace("/", "_").replace("\\", "_")
            buffer_path = str(_PROJECT_ROOT / "data" / f"remote_buffer_{safe_name}.json")
        else:
            buffer_path = os.path.join(os.path.dirname(log_path), "log_buffer.json")

        buffer = LogBuffer(buffer_path, max_size=windows["buffer_size"])

        new_snapshots = parse_log_to_snapshots(lines, format_config)
        if new_snapshots:
            existing = set(buffer.data.keys())
            new_snapshots = {k: v for k, v in new_snapshots.items() if k not in existing}
            buffer.update(new_snapshots)

        availability = check_data_availability(buffer.size(), windows)
        parse_result = compute_all(buffer, windows, format_config, total_iterations)
        parse_result["meta"]["data_availability"] = availability
        parse_result["_buffer_data"] = {str(k): v for k, v in buffer.data.items()}

        # ── 3. 保存快照 ──
        snapshot_dir = _PROJECT_ROOT / "data" / "snapshots" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(snapshot_dir, exist_ok=True)
        iter_num = parse_result.get("meta", {}).get("latest_iteration", "unknown")
        snapshot_path = os.path.join(snapshot_dir, f"iter{iter_num}.json")
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(parse_result, f, indent=2, ensure_ascii=False)

        # ── 4. 深度分析 ──
        buffer_data = parse_result["_buffer_data"]
        all_fields = list(buffer_data[next(iter(buffer_data.keys()))].keys())

        forecast_fields = [
            "Metrics/base_velocity/error_vel_xy",
            "Curriculum/terrain_levels",
            "Episode_Reward/feet_gait",
            "Episode_Reward/track_lin_vel_xy_exp",
        ]
        forecast_fields = [f for f in forecast_fields if f in all_fields]

        deep_result = {
            "change_points": compute_all_change_points(buffer_data, all_fields),
            "event_links": compute_all_event_links(buffer_data, all_fields),
            "anomalies": compute_anomalies(buffer_data, all_fields),
            "trend_forecast": compute_trend_forecast(buffer_data, forecast_fields),
            "before_after": None,
        }

        if data_dir:
            current = _load_snapshot(snapshot_path)
            previous = _find_previous_snapshot(data_dir, snapshot_path)
            if previous:
                deep_result["before_after"] = compute_before_after(
                    current, previous, all_fields
                )

        # ── 5. 标记 rebuttal 闸门 ──
        result = {
            "summary": parse_result.get("summary", {}),
            "snapshot_path": snapshot_path,
        }
        if deep:
            result["deep"] = deep_result

        if _control is not None:
            _control["_pending_rebuttal"] = True

        return _clean_for_json(result)

    except Exception as e:
        import traceback
        return {"error": f"analyze_training failed: {type(e).__name__}: {e}\n{traceback.format_exc()}"}
