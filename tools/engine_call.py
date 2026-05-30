"""
engine_call.py — 调用 engine1 运动学特征解析器
"""
import sys
import os
from datetime import datetime

# 确保能 import engine1
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from engine1.engine import parse

TOOL = {
    "name": "engine_call",
    "description": "调用 engine1 运动学特征解析器，分析 CSV 中的运动学数据，返回特征报告。",
    "parameters": {
        "csv_path": {"type": "str", "required": True, "desc": "my_play.py 采集的 CSV 文件路径（绝对路径或相对项目根目录）"},
    }
}

def execute(csv_path: str) -> dict:
    try:
        report = parse(csv_path)
        # 持久化到 data/sim_analysis/
        base = os.path.splitext(os.path.basename(csv_path))[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(_project_root, "data", "sim_analysis")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{base}_{ts}_features.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        return {
            "output_path": out_path,
            "error": None,
        }
    except Exception as e:
        return {"report": "", "output_path": "", "error": f"engine_call 失败: {type(e).__name__}: {e}"}
