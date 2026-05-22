"""
parse_log.py -- 训练日志解析与数值特征计算

从训练日志中增量解析迭代数据，维护滑动窗口缓冲区，
计算趋势、相关性、因果链、聚类健康度等特征，供 Agent 诊断使用。

支持本地文件和远程 SSH 文件两种模式。
远程模式自动从 config/ssh.json 读取凭据。

MCP Tool: parse_training_log
"""
import json
import os
import re
import sys
import argparse
import tempfile
import numpy as np
from pathlib import Path
from collections import deque
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.linear_model import LinearRegression

# ============================================================
# MCP Tool Definition
# ============================================================

TOOL = {
    "name": "parse_training_log",
    "description": (
        "解析四足机器人强化学习训练日志，返回当前训练状态的结构化分析。"
        "包含趋势、收敛、奖励结构、相关性、因果链评分、聚类健康度等特征。"
        "Agent 应在每次需要了解训练进展时调用此工具。"
    ),
    "parameters": {
        "log_path": {
            "type": "str", "required": True,
            "desc": "训练日志文件路径。本地路径或远程绝对路径。"
        },
        "total_iterations": {
            "type": "int", "required": False,
            "desc": "训练总轮数。不传则从日志中自动提取"
        },
        "log_format": {
            "type": "str", "required": False,
            "desc": "日志格式配置名称。内置: 'isaac_rl'。默认 'isaac_rl'"
        },
        "output": {
            "type": "str", "required": False,
            "desc": "可选，将结果写入指定 JSON 文件路径"
        },
        "remote": {
            "type": "bool", "required": False,
            "desc": "日志在远程服务器上。true 时自动从 config/ssh.json 读取凭据，SSH 拉取日志"
        }
    }
}

# ============================================================
# Configuration
# ============================================================

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config", "log_formats")
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_ssh_config():
    ssh_cfg_path = _PROJECT_ROOT / "config" / "ssh.json"
    if ssh_cfg_path.exists():
        with open(ssh_cfg_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def _load_builtin_formats():
    formats = {}
    if not os.path.isdir(_CONFIG_DIR):
        return formats
    for fname in os.listdir(_CONFIG_DIR):
        if fname.endswith(".json"):
            name = fname[:-5]
            path = os.path.join(_CONFIG_DIR, fname)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    formats[name] = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
    return formats


BUILTIN_FORMATS = _load_builtin_formats()

# ============================================================
# Constants
# ============================================================

TAIL_LINES = 20_000
MAX_BUFFER_SIZE_DEFAULT = 500
MIN_BUFFER_SIZE = 200
DEFAULT_TOTAL_ITER = 10_000
MAX_OUTPUT_CHARS = 500_000

# ============================================================
# Remote Log Fetching
# ============================================================


def _fetch_remote_log(log_path, n_lines=20000):
    ssh_cfg = _load_ssh_config()
    if not ssh_cfg:
        raise RuntimeError("remote=true 但 config/ssh.json 不存在或为空")

    import paramiko
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=ssh_cfg.get("ssh_host", ""),
        port=int(ssh_cfg.get("ssh_port", 22)),
        username=ssh_cfg.get("ssh_user", "root"),
        password=ssh_cfg.get("ssh_pass", ""),
        timeout=15
    )
    stdin, stdout, stderr = client.exec_command(
        f"tail -{n_lines} {log_path}", timeout=30
    )
    content = stdout.read().decode("utf-8", errors="replace")
    client.close()

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False, encoding='utf-8')
    tmp.write(content)
    tmp.close()
    return tmp.name


# ============================================================
# Log Parsing
# ============================================================


def load_format_config(log_format):
    if log_format in BUILTIN_FORMATS:
        return BUILTIN_FORMATS[log_format]
    if os.path.exists(log_format):
        with open(log_format, 'r', encoding='utf-8') as f:
            return json.load(f)
    raise ValueError(f"Unknown log_format: {log_format}")


def tail_log(log_path, n_lines):
    try:
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"日志文件不存在: {log_path}")
    return lines[-n_lines:] if len(lines) > n_lines else lines


def extract_total_iterations(lines, format_config):
    pattern = format_config["iteration_pattern"]
    for line in lines:
        match = re.search(pattern, line)
        if match and len(match.groups()) >= 2:
            return int(match.group(2))
    return None


def parse_field_line(line, format_config):
    line = line.strip()
    if not line or line.startswith('#'):
        return None
    for kw in format_config["exclude_keywords"]:
        if kw in line:
            return None
    match = re.match(format_config["field_pattern"], line)
    if not match:
        return None
    key = match.group(1).strip()
    value_str = match.group(2).strip()
    try:
        value = float(value_str)
    except ValueError:
        return None
    return key, value


def parse_log_to_snapshots(lines, format_config):
    snapshots = {}
    current_iter = None
    current_data = {}
    iter_pattern = format_config["iteration_pattern"]

    for line in lines:
        iter_match = re.search(iter_pattern, line)
        if iter_match:
            if current_iter is not None and current_data:
                snapshots[current_iter] = current_data
            current_iter = int(iter_match.group(1))
            current_data = {}
            continue
        if current_iter is not None:
            result = parse_field_line(line, format_config)
            if result:
                key, value = result
                current_data[key] = value
    if current_iter is not None and current_data:
        snapshots[current_iter] = current_data
    return snapshots


# ============================================================
# Buffer Management
# ============================================================


class LogBuffer:
    def __init__(self, buffer_path, max_size=MAX_BUFFER_SIZE_DEFAULT):
        self.buffer_path = buffer_path
        self.max_size = max_size
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.buffer_path):
            try:
                with open(self.buffer_path, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                return {int(k): v for k, v in raw.items()}
            except (json.JSONDecodeError, ValueError):
                pass
        return {}

    def save(self):
        os.makedirs(os.path.dirname(self.buffer_path) or ".", exist_ok=True)
        with open(self.buffer_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def update(self, new_snapshots: dict):
        if not new_snapshots:
            return
        self.data.update(new_snapshots)
        if len(self.data) > self.max_size:
            sorted_keys = sorted(self.data.keys())
            to_delete = sorted_keys[:-self.max_size]
            for k in to_delete:
                del self.data[k]
        self.save()

    def get_series(self, field_name):
        sorted_items = sorted(self.data.items(), key=lambda x: x[0])
        return [snapshot.get(field_name) for snapshot in [v for _, v in sorted_items]]

    def get_latest_snapshot(self):
        if not self.data:
            return {}
        latest_iter = max(self.data.keys())
        return self.data[latest_iter]

    def get_latest_iteration(self):
        if not self.data:
            return None
        return max(self.data.keys())

    def size(self):
        return len(self.data)


# ============================================================
# Adaptive Windows
# ============================================================


def compute_adaptive_windows(total_iterations):
    def scale(ratio, min_val, max_val):
        return int(np.clip(total_iterations * ratio, min_val, max_val))

    return {
        "buffer_size": scale(0.03, MIN_BUFFER_SIZE, 500),
        "trend_window": scale(0.005, 30, 80),
        "monotonicity_window": scale(0.003, 20, 50),
        "significance_window": scale(0.005, 30, 80),
        "volatility_recent": scale(0.003, 20, 50),
        "volatility_reference": scale(0.006, 30, 80),
        "convergence_window": scale(0.005, 30, 80),
        "outlier_check_n": scale(0.001, 5, 15),
        "stagnation_recent": scale(0.005, 30, 80),
        "stagnation_reference": scale(0.005, 30, 80),
        "stagnation_gap": scale(0.002, 10, 30),
        "correlation_window": scale(0.01, 50, 150),
        "leadlag_max_lag": scale(0.002, 10, 30),
        "efficiency_window": scale(0.005, 30, 80),
        "divergence_window": scale(0.005, 30, 80),
        "early_stage_ratio": 0.15,
    }


# ============================================================
# Feature Computation
# ============================================================


def _clean_series(series):
    return [v for v in series if v is not None]


def align_series(series_a, series_b):
    min_len = min(len(series_a), len(series_b))
    a_tail = series_a[-min_len:]
    b_tail = series_b[-min_len:]
    result_a, result_b = [], []
    for va, vb in zip(a_tail, b_tail):
        if va is not None and vb is not None:
            result_a.append(va)
            result_b.append(vb)
    return result_a, result_b


def compute_slope(series):
    series = _clean_series(series)
    if len(series) < 3:
        return None
    x = np.arange(len(series)).reshape(-1, 1)
    y = np.array(series)
    model = LinearRegression().fit(x, y)
    return float(model.coef_[0])


def compute_acceleration(series):
    series = _clean_series(series)
    if len(series) < 4:
        return None
    diffs = np.diff(series, n=2)
    return float(np.mean(diffs))


def compute_monotonicity(series):
    series = _clean_series(series)
    if len(series) < 3:
        return None
    diffs = np.diff(series)
    pos = np.sum(diffs > 0)
    neg = np.sum(diffs < 0)
    total = pos + neg
    if total == 0:
        return 0.0
    return float((pos - neg) / total)


def test_trend_significance(series):
    series = _clean_series(series)
    if len(series) < 4:
        return None
    n = len(series)
    s = 0
    for i in range(n):
        for j in range(i + 1, n):
            s += np.sign(series[j] - series[i])
    var_s = n * (n - 1) * (2 * n + 5) / 18
    if var_s <= 0:
        return None
    z = (s - 1) / np.sqrt(var_s) if s > 0 else (s + 1) / np.sqrt(var_s)
    return {
        "z_score": float(z),
        "significant": bool(abs(z) > 1.96),
        "p_approx": "< 0.05" if abs(z) > 1.96 else ">= 0.05",
    }


def compute_volatility(series):
    series = _clean_series(series)
    if len(series) < 3:
        return None
    arr = np.array(series)
    mean = np.mean(arr)
    if abs(mean) < 1e-10:
        return None
    return float(np.std(arr) / abs(mean))


def check_convergence(series, window, field_name=None):
    series = _clean_series(series)
    if len(series) < window:
        return None
    thresholds = _get_field_thresholds(field_name or "")
    recent = series[-window:]
    arr = np.array(recent)
    mean = np.mean(arr)
    if abs(mean) < 1e-10:
        return None
    cv = float(np.std(arr) / abs(mean))
    return {
        "converged": bool(cv < thresholds["convergence_cv"]),
        "cv": cv,
        "threshold": thresholds["convergence_cv"],
        "mean": float(mean),
        "std": float(np.std(arr)),
    }


def detect_outliers(series, n):
    series = _clean_series(series)
    if len(series) < n:
        return []
    recent = series[-n:]
    arr = np.array(recent)
    mean = np.mean(arr)
    std = np.std(arr)
    if std < 1e-10:
        return []
    outliers = []
    for i, v in enumerate(recent):
        z = abs((v - mean) / std)
        if z > 3:
            outliers.append({"index": len(series) - n + i, "value": v, "z_score": float(z)})
    return outliers


def detect_stagnation(series, recent_window, reference_window, gap, field_name=None):
    series = _clean_series(series)
    if len(series) < recent_window + reference_window + gap:
        return None
    thresholds = _get_field_thresholds(field_name or "")
    recent = series[-recent_window:]
    reference = series[-(recent_window + gap + reference_window):-(recent_window + gap)]
    recent_mean = float(np.mean(recent))
    ref_mean = float(np.mean(reference))
    change_pct = (recent_mean - ref_mean) / (abs(ref_mean) + 1e-10)
    return {
        "stagnant": (abs(change_pct) < thresholds["stagnation_change_pct"]),
        "change_pct": change_pct,
        "threshold": thresholds["stagnation_change_pct"],
        "recent_mean": recent_mean,
        "ref_mean": ref_mean,
    }


def compute_correlation(series_a, series_b):
    a, b = align_series(series_a, series_b)
    if len(a) < 5:
        return None
    try:
        pearson_r, pearson_p = pearsonr(a, b)
        spearman_r, spearman_p = spearmanr(a, b)
        kendall_t, kendall_p = kendalltau(a, b)
        return {
            "pearson": {"r": float(pearson_r), "p": float(pearson_p)},
            "spearman": {"r": float(spearman_r), "p": float(spearman_p)},
            "kendall": {"tau": float(kendall_t), "p": float(kendall_p)},
        }
    except Exception:
        return None


def detect_correlation_shift(series_a, series_b, window):
    a = _clean_series(series_a)
    b = _clean_series(series_b)
    if len(a) < window * 2 or len(b) < window * 2:
        return None
    min_len = min(len(a), len(b))
    half = min_len // 2
    try:
        r1, _ = pearsonr(a[:half], b[:half])
        r2, _ = pearsonr(a[-half:], b[-half:])
        return {
            "first_half_r": float(r1),
            "second_half_r": float(r2),
            "shift": float(r2 - r1),
        }
    except Exception:
        return None


def compute_lead_lag(series_a, series_b, max_lag):
    a = _clean_series(series_a)
    b = _clean_series(series_b)
    if len(a) < 10 or len(b) < 10:
        return None
    min_len = min(len(a), len(b))
    a, b = a[-min_len:], b[-min_len:]
    best_lag = 0
    best_corr = 0
    for lag in range(1, min(max_lag, min_len // 2) + 1):
        try:
            r, _ = pearsonr(a[:-lag], b[lag:])
            if abs(r) > abs(best_corr):
                best_corr = r
                best_lag = lag
            r, _ = pearsonr(a[lag:], b[:-lag])
            if abs(r) > abs(best_corr):
                best_corr = r
                best_lag = -lag
        except Exception:
            continue
    return {"best_lag": best_lag, "best_corr": float(best_corr)}


def compute_reward_structure(snapshot, clusters):
    total_reward = snapshot.get("Mean reward", 0)
    if total_reward is None or total_reward == 0:
        return None
    structure = {}
    for cluster_name, cluster_def in clusters.items():
        cluster_sum = 0.0
        available = 0
        for member in cluster_def["members"]:
            val = snapshot.get(member)
            if val is not None:
                cluster_sum += val
                available += 1
        if available > 0:
            structure[cluster_name] = {
                "sum": cluster_sum,
                "pct_of_total": cluster_sum / total_reward,
                "available_members": available,
                "direction": cluster_def["direction"],
                "description": cluster_def["description"],
            }
    return structure


def compute_penalty_concentration(snapshot, clusters):
    constraint = clusters.get("constraint_violation", {})
    members = constraint.get("members", [])
    penalty_sum = 0.0
    for m in members:
        val = snapshot.get(m)
        if val is not None and val < 0:
            penalty_sum += abs(val)
    total_neg = sum(abs(v) for v in snapshot.values() if isinstance(v, (int, float)) and v < 0)
    if total_neg == 0:
        return None
    return {
        "constraint_penalty_sum": penalty_sum,
        "total_negative_sum": total_neg,
        "concentration": penalty_sum / total_neg,
    }


def find_dominant_penalty(snapshot):
    best_key, best_val = None, 0
    for k, v in snapshot.items():
        if isinstance(v, (int, float)) and v < 0 and abs(v) > abs(best_val):
            best_key, best_val = k, v
    if best_key is None:
        return None
    return {"field": best_key, "value": best_val}


def compute_reward_efficiency(series, window):
    series = _clean_series(series)
    if len(series) < window:
        return None
    recent = series[-window:]
    if len(recent) < 2:
        return None
    improvement = recent[-1] - recent[0]
    return {
        "improvement": float(improvement),
        "per_iteration": float(improvement / len(recent)),
        "window": window,
    }


def detect_divergence(series_a, series_b, window):
    a = _clean_series(series_a)
    b = _clean_series(series_b)
    if len(a) < window or len(b) < window:
        return None
    min_len = min(len(a), len(b))
    a_tail, b_tail = a[-min_len:], b[-min_len:]
    gap = [abs(a_tail[i] - b_tail[i]) for i in range(min_len)]
    recent_gap = gap[-window:]
    gap_slope = compute_slope(recent_gap)
    a_arr = np.array(a_tail)
    b_arr = np.array(b_tail)
    a_z = (a_arr - np.mean(a_arr)) / (np.std(a_arr) + 1e-10)
    b_z = (b_arr - np.mean(b_arr)) / (np.std(b_arr) + 1e-10)
    recent_a_z = a_z[-window:]
    recent_b_z = b_z[-window:]
    both_high = np.sum((recent_a_z > 0.5) & (recent_b_z > 0.5))
    return {
        "diverging": bool(gap_slope > 0) if gap_slope is not None else None,
        "gap_slope": gap_slope,
        "current_gap": float(recent_gap[-1]),
        "both_high_ratio": float(both_high / len(recent_a_z)),
        "both_high_count": int(both_high),
    }


def classify_stage(iteration, total_iter, terrain_level, reward_series=None):
    progress = iteration / total_iter if total_iter else 0
    if progress < 0.15:
        stage = "early_exploration"
    elif terrain_level < 3:
        stage = "basic_skill"
    elif terrain_level < 6:
        stage = "curriculum_advancing"
    elif terrain_level < 8:
        stage = "complex_terrain"
    else:
        stage = "fine_tuning"
    return {"stage": stage, "progress": progress, "terrain_level": terrain_level}


def compute_partial_correlation(series_a, series_b, series_control):
    a = _clean_series(series_a)
    b = _clean_series(series_b)
    c = _clean_series(series_control)
    if len(a) < 10 or len(b) < 10 or len(c) < 10:
        return None
    min_len = min(len(a), len(b), len(c))
    a, b, c = a[-min_len:], b[-min_len:], c[-min_len:]
    try:
        def _residual(x, control):
            x_arr = np.array(x)
            c_arr = np.array(control).reshape(-1, 1)
            model = LinearRegression().fit(c_arr, x_arr)
            return x_arr - model.predict(c_arr)

        res_a = _residual(a, c)
        res_b = _residual(b, c)
        r, p = pearsonr(res_a, res_b)
        return {"partial_r": float(r), "p": float(p)}
    except Exception:
        return None


def score_causal_chains(buffer, format_config):
    rules = format_config.get("partial_correlation_rules", [])
    if not rules:
        return {}
    scores = {}
    for rule in rules:
        if len(rule) < 3:
            continue
        a_field, b_field, c_field = rule[0], rule[1], rule[2]
        series_a = buffer.get_series(a_field)
        series_b = buffer.get_series(b_field)
        series_c = buffer.get_series(c_field)
        pc = compute_partial_correlation(series_a, series_b, series_c)
        if pc is not None:
            zero_order = compute_correlation(series_a, series_b)
            zero_r = zero_order["pearson"]["r"] if zero_order else None
            key = f"{a_field} | {c_field} -> {b_field}"
            scores[key] = {
                "partial_r": pc["partial_r"],
                "p": pc["p"],
                "zero_order_r": zero_r,
            }
    return scores


def compute_cluster_health(buffer, clusters, window):
    health = {}
    for cluster_name, cluster_def in clusters.items():
        members = cluster_def["members"]
        slopes = []
        for member in members:
            series = buffer.get_series(member)
            s = compute_slope(series[-window:] if len(series) > window else series)
            if s is not None:
                slopes.append(s)
        if not slopes:
            health[cluster_name] = {
                "avg_slope": None,
                "pos_ratio": None,
                "member_count": len(members),
                "description": cluster_def["description"],
            }
            continue
        avg_slope = float(np.mean(slopes))
        pos_ratio = sum(1 for s in slopes if s > 0) / len(slopes)
        health[cluster_name] = {
            "avg_slope": avg_slope,
            "pos_ratio": pos_ratio,
            "member_count": len(members),
            "description": cluster_def["description"],
        }
    return health


def check_data_availability(buffer_size, windows):
    return {
        "trend": buffer_size >= windows["trend_window"],
        "monotonicity": buffer_size >= windows["monotonicity_window"],
        "significance": buffer_size >= windows["significance_window"],
        "volatility": buffer_size >= (windows["volatility_recent"] + windows["volatility_reference"]),
        "convergence": buffer_size >= windows["convergence_window"],
        "outliers": buffer_size >= 30,
        "stagnation": buffer_size >= (windows["stagnation_recent"] * 2 + windows["stagnation_gap"]),
        "correlation": buffer_size >= windows["correlation_window"],
        "correlation_shift": buffer_size >= windows["correlation_window"] * 2,
        "lead_lag": buffer_size >= windows["correlation_window"],
        "efficiency": buffer_size >= windows["efficiency_window"],
        "divergence": buffer_size >= windows["divergence_window"],
        "causal_partial": buffer_size >= windows["correlation_window"],
        "causal_chains": buffer_size >= windows["correlation_window"],
        "cluster_health": buffer_size >= windows["trend_window"],
    }


def _get_field_thresholds(field_name):
    lower = field_name.lower()
    if "entropy" in lower:
        return {"convergence_cv": 0.1, "stagnation_change_pct": 0.02}
    if "reward" in lower or "track" in lower or "feet" in lower or "upward" in lower:
        return {"convergence_cv": 0.08, "stagnation_change_pct": 0.015}
    if "penalty" in lower or "constraint" in lower or "contact" in lower:
        return {"convergence_cv": 0.15, "stagnation_change_pct": 0.03}
    if "error" in lower or "noise" in lower:
        return {"convergence_cv": 0.1, "stagnation_change_pct": 0.02}
    if "terrain" in lower:
        return {"convergence_cv": 0.2, "stagnation_change_pct": 0.05}
    return {"convergence_cv": 0.05, "stagnation_change_pct": 0.01}


# ============================================================
# Main Computation
# ============================================================


def compute_all(buffer, windows, format_config, total_iter, availability=None):
    latest = buffer.get_latest_snapshot()
    series = {}
    for field in format_config.get("trend_fields", []):
        s = buffer.get_series(field)
        s_clean = [v for v in s if v is not None]
        if s_clean:
            series[field] = s_clean

    terrain_series = buffer.get_series("Curriculum/terrain_levels")
    terrain_latest = terrain_series[-1] if terrain_series else 0
    stage_result = classify_stage(
        buffer.get_latest_iteration() or 0,
        total_iter,
        terrain_latest,
        buffer.get_series("Mean reward")
    )

    trends = {}
    for field, s in series.items():
        if len(s) < 3:
            continue
        slope = compute_slope(s)
        accel = compute_acceleration(s)
        mono = compute_monotonicity(s)
        sig = test_trend_significance(s)
        conv = check_convergence(s, windows["convergence_window"], field_name=field)
        vol = compute_volatility(s[-windows["volatility_recent"]:]) if len(s) >= windows["volatility_recent"] else None
        outliers = detect_outliers(s, windows["outlier_check_n"])
        stag = detect_stagnation(s, windows["stagnation_recent"], windows["stagnation_reference"], windows["stagnation_gap"], field_name=field)
        trends[field] = {
            "slope": slope,
            "acceleration": accel,
            "monotonicity": mono,
            "significance_z": sig,
            "convergence": conv,
            "volatility": vol,
            "outliers": outliers,
            "stagnation": stag,
        }

    correlations = {}
    for pair in format_config.get("correlation_pairs", []):
        if len(pair) < 2:
            continue
        a_field, b_field = pair[0], pair[1]
        s_a = buffer.get_series(a_field)
        s_b = buffer.get_series(b_field)
        corr = compute_correlation(s_a, s_b)
        shift = detect_correlation_shift(s_a, s_b, windows["correlation_window"] // 2)
        leadlag = compute_lead_lag(s_a, s_b, windows["leadlag_max_lag"])
        div = detect_divergence(s_a, s_b, windows["divergence_window"])
        key = f"{a_field} <-> {b_field}"
        correlations[key] = {
            "correlation": corr,
            "shift": shift,
            "lead_lag": leadlag,
            "divergence": div,
        }

    clusters = format_config.get("clusters", {})
    structure = compute_reward_structure(latest, clusters)
    penalty_conc = compute_penalty_concentration(latest, clusters)
    dominant_penalty = find_dominant_penalty(latest)
    efficiency = compute_reward_efficiency(series.get("Mean reward", []), windows["efficiency_window"])

    divergence = {}
    for pair in format_config.get("correlation_pairs", []):
        if len(pair) < 2:
            continue
        a_field, b_field = pair[0], pair[1]
        s_a = buffer.get_series(a_field)
        s_b = buffer.get_series(b_field)
        div = detect_divergence(s_a, s_b, windows["divergence_window"])
        if div is not None:
            key = f"{a_field} <-> {b_field}"
            divergence[key] = div

    causal_analysis = {
        "partial_correlations": score_causal_chains(buffer, format_config),
    }

    cluster_health = compute_cluster_health(buffer, clusters, windows["trend_window"])

    # ================================================================
    # Summary (返回给 Agent)
    # ================================================================

    # 所有惩罚项
    all_penalties = {}
    for k, v in latest.items():
        if isinstance(v, (int, float)) and v < 0:
            all_penalties[k] = v
    all_penalties = dict(sorted(all_penalties.items(), key=lambda x: x[1]))

    # 所有正奖励项
    all_rewards = {}
    for k, v in latest.items():
        if isinstance(v, (int, float)) and v > 0:
            all_rewards[k] = v
    all_rewards = dict(sorted(all_rewards.items(), key=lambda x: x[1], reverse=True))

    # 正负比
    pos_sum = sum(v for v in latest.values() if isinstance(v, (int, float)) and v > 0)
    neg_sum = sum(abs(v) for v in latest.values() if isinstance(v, (int, float)) and v < 0)
    pos_neg_ratio = round(pos_sum / neg_sum, 3) if neg_sum > 0 else None

    # 主导惩罚
    dom_field = None
    dom_val = 0
    if dominant_penalty:
        dom_field = dominant_penalty.get("field")
        dom_val = abs(dominant_penalty.get("value", 0))
    total_neg = sum(abs(v) for v in latest.values() if isinstance(v, (int, float)) and v < 0)
    dominant_penalty_share = round(dom_val / total_neg, 3) if total_neg > 0 else None

    # 约束集中度
    conc = 0
    if penalty_conc:
        conc = penalty_conc.get("concentration", 0)

    # 关键指标 volatility
    vol_fields = [
        "Metrics/base_velocity/error_vel_xy",
        "Curriculum/terrain_levels",
        "Episode_Reward/feet_gait",
        "Episode_Reward/track_lin_vel_xy_exp",
    ]
    volatility = {}
    for f in vol_fields:
        t = trends.get(f)
        if t and t.get("volatility") is not None:
            volatility[f] = round(t["volatility"], 4)

    # 响应延迟
    response_lags = {}
    for key, val in correlations.items():
        if val and val.get("lead_lag") and val["lead_lag"].get("best_lag") is not None:
            response_lags[key] = val["lead_lag"]["best_lag"]

    # 偏相关
    partial_correlations_summary = {}
    for k, v in causal_analysis.get("partial_correlations", {}).items():
        partial_correlations_summary[k] = {
            "partial_r": v["partial_r"],
            "p": v["p"],
            "zero_order_r": v["zero_order_r"],
        }

    # 斜率汇总
    slope_fields = [
        "Episode_Reward/track_lin_vel_xy_exp",
        "Metrics/base_velocity/error_vel_xy",
        "Curriculum/terrain_levels",
        "Episode_Reward/feet_gait",
        "Episode_Reward/joint_pos_penalty",
        "Episode_Reward/undesired_contacts",
        "Episode_Reward/joint_acc_l2",
        "Episode_Reward/action_rate_l2",
    ]
    slopes = {}
    for f in slope_fields:
        t = trends.get(f)
        if t and t.get("slope") is not None:
            slopes[f] = round(t["slope"], 6)

    # 聚类斜率
    cluster_slopes = {}
    for k, v in cluster_health.items():
        if v.get("avg_slope") is not None:
            cluster_slopes[k] = round(v["avg_slope"], 6)

    # 构建 summary
    summary = {
        "iter": buffer.get_latest_iteration(),
        "stage": stage_result["stage"],
        "progress": stage_result["progress"],
        "snapshot": {
            "track_reward": latest.get("Episode_Reward/track_lin_vel_xy_exp"),
            "error_vel_xy": latest.get("Metrics/base_velocity/error_vel_xy"),
            "error_vel_yaw": latest.get("Metrics/base_velocity/error_vel_yaw"),
            "terrain": latest.get("Curriculum/terrain_levels"),
            "time_out": latest.get("Episode_Termination/time_out"),
            "feet_gait": latest.get("Episode_Reward/feet_gait"),
            "upward": latest.get("Episode_Reward/upward"),
            "entropy": latest.get("Mean entropy loss"),
            "action_noise_std": latest.get("Mean action noise std"),
        },
        "all_penalties": all_penalties,
        "all_rewards": all_rewards,
        "slopes": slopes,
        "volatility": volatility,
        "ratios": {
            "pos_neg": pos_neg_ratio,
            "dominant_penalty_share": dominant_penalty_share,
            "constraint_concentration": round(conc, 3),
        },
        "dominant_penalty": dom_field,
        "response_lags": response_lags,
        "cluster_slopes": cluster_slopes,
        "partial_correlations": partial_correlations_summary,
    }

    # ================================================================
    # 完整 JSON（写入快照目录）
    # ================================================================
    output = {
        "meta": {
            "buffer_size": buffer.size(),
            "latest_iteration": buffer.get_latest_iteration(),
            "total_iterations": total_iter,
            "stage": stage_result,
            "data_availability": availability or {},
        },
        "latest_snapshot": latest,
        "cluster_health": cluster_health,
        "trends": trends,
        "correlations": correlations,
        "reward_structure": {
            "decomposition": structure,
            "penalty_concentration": penalty_conc,
            "dominant_penalty": dominant_penalty,
            "efficiency": efficiency,
        },
        "divergence": divergence,
        "causal_analysis": causal_analysis,
        "summary": summary,
    }

    return output


# ============================================================
# MCP Tool Entry Point
# ============================================================


def execute(log_path: str, total_iterations: int = None, log_format: str = "isaac_rl",
            output: str = None, remote: bool = False) -> dict:
    try:
        if remote:
            log_path = _fetch_remote_log(log_path, TAIL_LINES)

        format_config = load_format_config(log_format)
        lines = tail_log(log_path, TAIL_LINES)

        if total_iterations is None:
            total_iterations = extract_total_iterations(lines, format_config)
        if total_iterations is None:
            total_iterations = DEFAULT_TOTAL_ITER

        windows = compute_adaptive_windows(total_iterations)

        if remote:
            safe_name = log_path.replace("/", "_").replace("\\", "_")
            buffer_path = _PROJECT_ROOT / "data" / f"remote_buffer_{safe_name}.json"
            buffer_path = str(buffer_path)
        else:
            buffer_path = os.path.join(os.path.dirname(log_path), "log_buffer.json")

        buffer = LogBuffer(buffer_path, max_size=windows["buffer_size"])

        new_snapshots = parse_log_to_snapshots(lines, format_config)
        if new_snapshots:
            existing = set(buffer.data.keys())
            new_snapshots = {k: v for k, v in new_snapshots.items() if k not in existing}
            buffer.update(new_snapshots)

        availability = check_data_availability(buffer.size(), windows)
        result = compute_all(buffer, windows, format_config, total_iterations)
        result["meta"]["data_availability"] = availability

        # 将 buffer 原始时序数据写入快照（deep_dig 需要）
        result["_buffer_data"] = {str(k): v for k, v in buffer.data.items()}

        # 完整 JSON 写入快照目录
        from datetime import datetime
        snapshot_dir = _PROJECT_ROOT / "data" / "snapshots" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(snapshot_dir, exist_ok=True)
        iter_num = result.get("meta", {}).get("latest_iteration", "unknown")
        snapshot_path = os.path.join(snapshot_dir, f"iter{iter_num}.json")
        with open(snapshot_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        if output:
            out_dir = os.path.dirname(output) or "."
            os.makedirs(out_dir, exist_ok=True)
            with open(output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        return result.get("summary", {})

    except Exception as e:
        return {"error": f"parse_training_log failed: {type(e).__name__}: {e}"}


# ============================================================
# CLI Entry Point
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="Parse training log and compute features")
    parser.add_argument("--log_path", required=True)
    parser.add_argument("--total_iterations", type=int, default=None)
    parser.add_argument("--log_format", default="isaac_rl")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--remote", action="store_true", default=False)
    args = parser.parse_args()

    try:
        result = execute(args.log_path, args.total_iterations, args.log_format,
                         args.output, args.remote)
        output_str = json.dumps(result, indent=2, ensure_ascii=False)
        if len(output_str) > MAX_OUTPUT_CHARS:
            output_str = output_str[:MAX_OUTPUT_CHARS] + "\n... [truncated]"
        print(output_str)
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()