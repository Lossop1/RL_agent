"""
deep_dig.py — 训练数据深度挖掘工具

从 parse_log 产出的完整 JSON 中提取时序模式：
- 变点检测：趋势结构性变化的位置
- 事件关联：指标间的时间延迟和因果线索
- 异常检测：单轮或短区间的异常模式
- 趋势预测：未来趋势的外推和收敛估计
- 效果量化：调参前后跨快照的统计检验

MCP Tool: deep_dig
"""
import json
import os
import sys
import argparse
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ============================================================
# MCP Tool Definition
# ============================================================

TOOL = {
    "name": "deep_dig",
    "description": (
        "从训练快照的完整时序数据中挖掘深层模式。"
        "包含变点检测、事件关联、异常标注、趋势预测、调参效果量化。"
        "只产出结构化数据，不做判断。"
    ),
    "parameters": {
        "snapshot_path": {
            "type": "str",
            "required": True,
            "desc": "当前快照的完整 JSON 文件路径"
        },
        "data_dir": {
            "type": "str",
            "required": False,
            "desc": "快照目录。需要效果量化时传入，从目录中找上一次快照做对比"
        }
    }
}

# ============================================================
# JSON 清洗
# ============================================================

def _clean_for_json(obj):
    """递归清洗 numpy 类型，确保 JSON 可序列化"""
    if isinstance(obj, dict):
        return {str(k): _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# ============================================================
# 辅助函数
# ============================================================

def _load_snapshot(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _find_previous_snapshot(data_dir, current_path):
    """在快照目录中找当前快照之前的最新快照"""
    if not os.path.isdir(data_dir):
        return None
    subdirs = sorted(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))],
        reverse=True
    )
    current_ts = os.path.basename(os.path.dirname(current_path))
    found_current = False
    for sub in subdirs:
        sub_path = os.path.join(data_dir, sub)
        for fname in sorted(os.listdir(sub_path), reverse=True):
            if fname.startswith("iter") and fname.endswith(".json"):
                if not found_current:
                    if sub == current_ts:
                        found_current = True
                    continue
                path = os.path.join(sub_path, fname)
                return _load_snapshot(path)
    return None


# ============================================================
# 一、变点检测 (PELT)
# ============================================================

def detect_change_points(series, pen=10):
    """
    PELT 算法检测均值变点。
    返回变点位置列表（每个变点后的第一个索引）。
    """
    n = len(series)
    if n < 20:
        return []

    arr = np.array(series, dtype=float)
    F = np.zeros(n + 1)
    F[0] = -pen
    cp = np.zeros(n + 1, dtype=int)

    for t in range(1, n + 1):
        best_cost = float("inf")
        best_s = 0
        for s in range(t):
            segment = arr[s:t]
            cost = np.sum((segment - np.mean(segment)) ** 2)
            total = F[s] + cost + pen
            if total < best_cost:
                best_cost = total
                best_s = s
        F[t] = best_cost
        cp[t] = best_s

    change_points = []
    t = n
    while t > 0:
        s = cp[t]
        if s > 0:
            change_points.append(s)
        t = s
    return sorted(change_points)


def compute_all_change_points(buffer_data, trend_fields):
    """对所有 trend_fields 做变点检测"""
    results = []
    for field in trend_fields:
        series = [snap.get(field) for snap in buffer_data.values()]
        series = [v for v in series if v is not None]
        if len(series) < 20:
            continue
        cps = detect_change_points(series)
        for cp in cps:
            before = np.mean(series[:cp])
            after = np.mean(series[cp:])
            if abs(before) > 1e-8:
                pct = (after - before) / abs(before) * 100
            else:
                pct = float("inf") if after != 0 else 0
            results.append({
                "metric": field,
                "iteration": int(cp),
                "before_mean": round(float(before), 4),
                "after_mean": round(float(after), 4),
                "change_pct": round(float(pct), 1),
            })
    return results


# ============================================================
# 二、事件关联 (交叉相关 + 格兰杰)
# ============================================================

def compute_cross_correlation(x, y, max_lag=30):
    """计算两个序列的交叉相关，返回最大相关和对应延迟"""
    n = len(x)
    best_lag, best_corr = 0, 0
    for lag in range(1, min(max_lag, n // 3) + 1):
        if lag < n:
            corr = np.corrcoef(x[:-lag], y[lag:])[0, 1]
            if not np.isnan(corr) and abs(corr) > abs(best_corr):
                best_corr, best_lag = corr, lag
    return best_lag, best_corr


def compute_granger_causality(x, y, max_lag=5):
    """
    简易格兰杰因果检验。
    """
    n = len(x)
    if n < max_lag + 10:
        return 0, 1.0

    y_target = np.array(y[max_lag:], dtype=float)
    X_restricted = np.column_stack(
        [np.array(y[max_lag - i - 1:n - i - 1], dtype=float) for i in range(max_lag)]
    )

    valid = ~np.isnan(X_restricted).any(axis=1) & ~np.isnan(y_target)
    if valid.sum() < 10:
        return 0, 1.0
    X_r = X_restricted[valid]
    y_t = y_target[valid]
    beta_r = np.linalg.lstsq(X_r, y_t, rcond=None)[0]
    resid_r = y_t - X_r @ beta_r
    ssr_r = np.sum(resid_r ** 2)

    X_unrestricted = np.column_stack([
        X_r,
        np.column_stack(
            [np.array(x[max_lag - i - 1:n - i - 1], dtype=float) for i in range(max_lag)]
        )[valid]
    ])
    beta_u = np.linalg.lstsq(X_unrestricted, y_t, rcond=None)[0]
    resid_u = y_t - X_unrestricted @ beta_u
    ssr_u = np.sum(resid_u ** 2)

    df1 = max_lag
    df2 = len(y_t) - 2 * max_lag
    if df2 <= 0 or ssr_u < 1e-10:
        return 0, 1.0
    f_stat = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
    p_val = 1.0 if f_stat <= 0 else np.exp(-f_stat / 2)
    return round(float(f_stat), 4), round(float(min(p_val, 1.0)), 4)


def compute_all_event_links(buffer_data, all_fields):
    """对所有指标对做交叉相关和格兰杰检验"""
    first_key = next(iter(buffer_data.keys()))
    first_snap = buffer_data[first_key]
    fields = [f for f in all_fields if f in first_snap]

    results = []
    n = len(fields)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            x = [snap.get(fields[i]) for snap in buffer_data.values()]
            y = [snap.get(fields[j]) for snap in buffer_data.values()]
            x = [v for v in x if v is not None]
            y = [v for v in y if v is not None]
            min_len = min(len(x), len(y))
            x, y = x[-min_len:], y[-min_len:]

            if len(x) < 30:
                continue

            lag, corr = compute_cross_correlation(x, y)
            if abs(corr) < 0.6:
                continue

            f_stat, p_val = compute_granger_causality(x, y)

            results.append({
                "source": fields[i],
                "target": fields[j],
                "lag": int(lag),
                "cross_correlation": round(float(corr), 4),
                "granger_f": f_stat,
                "granger_p": p_val,
            })

    results.sort(key=lambda r: abs(r["cross_correlation"]), reverse=True)
    return results[:20]


# ============================================================
# 三、异常检测 (孤立森林)
# ============================================================

def isolation_forest(data_matrix, n_estimators=100, contamination=0.05):
    """
    简易孤立森林实现。
    返回每个样本的异常分数（0-1，越高越异常）。
    """
    n_samples = data_matrix.shape[0]
    depths = np.zeros(n_samples)

    for _ in range(n_estimators):
        sample_size = min(256, n_samples)
        sample_idx = np.random.choice(n_samples, sample_size, replace=False)
        subset = data_matrix[sample_idx].copy()

        node_indices = sample_idx.copy()

        while len(node_indices) > 1 and subset.shape[0] > 1:
            feat = np.random.randint(data_matrix.shape[1])
            col = subset[:, feat]
            min_v, max_v = col.min(), col.max()
            if min_v >= max_v:
                break
            split = np.random.uniform(min_v, max_v)
            left = col <= split

            depths[node_indices] += 1

            if left.sum() >= len(left) - left.sum():
                node_indices = node_indices[left]
                subset = subset[left]
            else:
                node_indices = node_indices[~left]
                subset = subset[~left]

    avg_depth = depths / n_estimators
    score = 2.0 ** (-avg_depth / np.log2(max(2, sample_size)))
    return score


def compute_anomalies(buffer_data, all_fields):
    """对 buffer 中所有轮次做异常检测"""
    iterations = sorted(buffer_data.keys())
    first_snap = buffer_data[iterations[0]]
    fields = [f for f in all_fields if f in first_snap]

    matrix = []
    for it in iterations:
        row = [buffer_data[it].get(f, 0) for f in fields]
        matrix.append(row)
    matrix = np.array(matrix, dtype=float)

    if matrix.shape[0] < 10:
        return []

    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    std[std < 1e-8] = 1.0
    matrix_norm = (matrix - mean) / std

    scores = isolation_forest(matrix_norm)
    threshold = np.percentile(scores, 95)

    anomalies = []
    for idx, (it, score) in enumerate(zip(iterations, scores)):
        if score > threshold:
            row = matrix_norm[idx]
            contributions = []
            for fi, f in enumerate(fields):
                contributions.append({
                    "metric": f,
                    "z_score": round(float(abs(row[fi])), 2),
                })
            contributions.sort(key=lambda c: c["z_score"], reverse=True)
            anomalies.append({
                "iteration": str(it),
                "anomaly_score": round(float(score), 4),
                "top_contributors": contributions[:5],
            })

    return sorted(anomalies, key=lambda a: a["anomaly_score"], reverse=True)[:10]


# ============================================================
# 四、趋势预测 (Holt-Winters + 指数衰减拟合)
# ============================================================

def holt_winters_forecast(series, forecast_steps=50, seasonal_period=10):
    """
    Holt-Winters 加法模型。返回预测值和 95% 置信区间。
    """
    series = np.array(series, dtype=float)
    n = len(series)
    if n < seasonal_period * 2:
        return None

    level = np.mean(series[:seasonal_period])
    trend = (np.mean(series[seasonal_period:2 * seasonal_period]) -
             np.mean(series[:seasonal_period])) / seasonal_period
    seasonal = series[:seasonal_period] - level

    alpha, beta, gamma = 0.3, 0.1, 0.1

    for t in range(seasonal_period, n):
        old_level = level
        level = alpha * (series[t] - seasonal[t % seasonal_period]) + (1 - alpha) * (level + trend)
        trend = beta * (level - old_level) + (1 - beta) * trend
        seasonal[t % seasonal_period] = gamma * (series[t] - level) + (1 - gamma) * seasonal[t % seasonal_period]

    forecast = []
    current_level = level
    current_trend = trend
    for step in range(1, forecast_steps + 1):
        f = current_level + step * current_trend + seasonal[(n + step - 1) % seasonal_period]
        forecast.append(round(float(f), 4))

    residuals = series[seasonal_period:] - np.array([
        level + (i + 1) * trend + seasonal[(seasonal_period + i) % seasonal_period]
        for i in range(n - seasonal_period)
    ])
    resid_std = float(np.std(residuals))
    lower = [round(f - 1.96 * resid_std, 4) for f in forecast]
    upper = [round(f + 1.96 * resid_std, 4) for f in forecast]

    return {
        "forecast": forecast,
        "lower_bound": lower,
        "upper_bound": upper,
    }


def fit_exponential_decay(series):
    """
    拟合 y = a * exp(-t / tau) + c。
    """
    from scipy.optimize import curve_fit
    series = np.array(series, dtype=float)
    t = np.arange(len(series), dtype=float)

    def model(t, a, tau, c):
        return a * np.exp(-t / tau) + c

    try:
        p0 = [series[0] - series[-1], len(series) / 2, series[-1]]
        bounds = ([0, 1, -np.inf], [np.inf, len(series) * 2, np.inf])
        popt, _ = curve_fit(model, t, series, p0=p0, bounds=bounds, maxfev=5000)
        predicted = model(t, *popt)
        ss_res = np.sum((series - predicted) ** 2)
        ss_tot = np.sum((series - np.mean(series)) ** 2)
        r_squared = 1 - ss_res / (ss_tot + 1e-10)
        return {
            "convergence_target": round(float(popt[2]), 4),
            "time_constant": round(float(popt[1]), 4),
            "r_squared": round(float(r_squared), 4),
        }
    except Exception:
        return None


def compute_trend_forecast(buffer_data, forecast_fields):
    """对核心指标做趋势预测"""
    results = {}
    for field in forecast_fields:
        series = [snap.get(field) for snap in buffer_data.values()]
        series = [v for v in series if v is not None]
        if len(series) < 30:
            continue

        hw = holt_winters_forecast(series)
        decay = fit_exponential_decay(series)

        results[field] = {
            "holt_winters": hw,
            "exponential_decay": decay,
        }
    return results


# ============================================================
# 五、效果量化 (Mann-Whitney U)
# ============================================================

def compute_before_after(current_snapshot, previous_snapshot, all_fields):
    """
    对比两份快照中每个指标的最新 N 轮数据。
    """
    curr_buffer = current_snapshot.get("_buffer_data")
    prev_buffer = previous_snapshot.get("_buffer_data")
    if not curr_buffer or not prev_buffer:
        return None

    curr_iters = sorted(curr_buffer.keys())
    prev_iters = sorted(prev_buffer.keys())

    curr_count = len(curr_iters)
    prev_count = len(prev_iters)
    window = min(curr_count, prev_count, 100)

    results = {}
    for field in all_fields:
        curr_vals = [curr_buffer[it].get(field) for it in curr_iters[-window:]]
        prev_vals = [prev_buffer[it].get(field) for it in prev_iters[-window:]]
        curr_vals = [v for v in curr_vals if v is not None]
        prev_vals = [v for v in prev_vals if v is not None]

        if len(curr_vals) < 10 or len(prev_vals) < 10:
            continue

        try:
            u_stat, p_val = mannwhitneyu(curr_vals, prev_vals, alternative="two-sided")
            results[field] = {
                "before_mean": round(float(np.mean(prev_vals)), 4),
                "after_mean": round(float(np.mean(curr_vals)), 4),
                "u_statistic": round(float(u_stat), 2),
                "p_value": round(float(p_val), 4),
            }
        except Exception:
            continue

    return results


# ============================================================
# 主入口
# ============================================================

def execute(snapshot_path, data_dir=None):
    """
    读取快照完整 JSON，产出五类深度分析结果。
    """
    try:
        current = _load_snapshot(snapshot_path)
    except Exception as e:
        return {"error": f"读取快照失败: {e}"}

    buffer_data = current.get("_buffer_data")
    if not buffer_data:
        return {"error": "快照中缺少 _buffer_data。请确保 parse_log 在完整 JSON 中保存了 buffer 原始数据"}

    all_fields = list(buffer_data[next(iter(buffer_data.keys()))].keys())

    forecast_fields = [
        "Metrics/base_velocity/error_vel_xy",
        "Curriculum/terrain_levels",
        "Episode_Reward/feet_gait",
        "Episode_Reward/track_lin_vel_xy_exp",
    ]

    result = {
        "change_points": compute_all_change_points(buffer_data, all_fields),
        "event_links": compute_all_event_links(buffer_data, all_fields),
        "anomalies": compute_anomalies(buffer_data, all_fields),
        "trend_forecast": compute_trend_forecast(buffer_data, forecast_fields),
    }

    if data_dir:
        previous = _find_previous_snapshot(data_dir, snapshot_path)
        if previous:
            result["before_after"] = compute_before_after(current, previous, all_fields)
        else:
            result["before_after"] = None

    return _clean_for_json(result)


# ============================================================
# CLI Entry Point
# ============================================================

MAX_OUTPUT_CHARS = 200_000

def main():
    parser = argparse.ArgumentParser(description="Deep dig training data")
    parser.add_argument("--snapshot_path", required=True)
    parser.add_argument("--data_dir", default=None)
    args = parser.parse_args()

    try:
        result = execute(args.snapshot_path, args.data_dir)
        output_str = json.dumps(result, indent=2, ensure_ascii=False)
        if len(output_str) > MAX_OUTPUT_CHARS:
            output_str = output_str[:MAX_OUTPUT_CHARS] + "\n... [truncated]"
        print(output_str)
    except Exception as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()