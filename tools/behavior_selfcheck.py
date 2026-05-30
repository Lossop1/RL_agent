"""
behavior_selfcheck.py --- 行为自洽性检查模块（v2）

从仿真 rollout CSV 提取时序特征，执行三类检查：
  A. 物理自洽 — 从力学定律和 URDF 参数推出的固真/固假判断
  B. 模式自洽 — 此 rollout 内部的统计事实，不设绝对阈值，由 LLM 解释
  C. 衍生指标 — 纯统计量，供对比 URDF 推导的标准阈值

设计原则：
  - A 类每条判断可追溯到物理公式或 URDF 参数
  - B 类只报告"存在什么现象"，不替 LLM 做"好/坏"判断
  - C 类不给判断，只给数值
  - 所有时序分析基于 numpy/scipy 信号处理

MCP Tool: behavior_selfcheck
"""
import json
import os
import csv
import math
import numpy as np
from collections import defaultdict

TOOL = {
    "name": "behavior_selfcheck",
    "description": (
        "读取仿真 rollout CSV，执行行为自洽性检查。"
        "A类（物理自洽）：从力学定律和 URDF 参数推出的固真判断。"
        "B类（模式自洽）：rollout 内部统计事实，由 LLM 结合上下文解读。"
        "C类（衍生指标）：纯统计量，供对比 URDF 推导的标准。"
    ),
    "parameters": {
        "csv_path": {
            "type": "str", "required": True,
            "desc": "仿真 rollout CSV 文件路径"
        },
        "kinematics_source": {
            "type": "str", "required": False,
            "desc": (
                "运动学参数来源。可选：(1) robot_kinematics 输出的 JSON 路径，"
                "(2) 内置 'go2' 使用默认 Go2 参数"
            )
        },
    }
}

# ============================================================================
# 辅助函数
# ============================================================================

def _load_csv(csv_path: str) -> tuple[list[str], list[dict[str, str]]]:
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        rows = list(reader)
    return headers, rows


def _col_array(rows: list[dict], col: str) -> np.ndarray:
    return np.array([float(r[col]) for r in rows])


def _cols_arrays(rows: list[dict], prefix: str, n: int) -> np.ndarray:
    """Extract columns prefix_0 .. prefix_{n-1} as (n_frames, n) array."""
    result = []
    for i in range(n):
        key = f"{prefix}{i}"
        if key in rows[0]:
            result.append([float(r[key]) for r in rows])
    return np.array(result).T if result else np.zeros((len(rows), 0))


def _load_kinematics(source: str | None) -> dict:
    """加载运动学参数。"""
    if source and os.path.exists(source):
        with open(source, "r") as f:
            return json.load(f)
    # 默认 Go2 参数
    return {
        "standing_height": 0.38,
        "leg_length": 0.426,
        "thigh_length": 0.213,
        "calf_length": 0.213,
        "hip_spacing_y": 0.093,
        "hip_x_forward": 0.1934,
        "foot_radius": 0.022,
        "joint_limits": {
            "FL": {"hip": {"lower": -0.9425, "upper": 0.9425, "range": 1.885},
                   "thigh": {"lower": -1.4137, "upper": 3.1416, "range": 4.5553},
                   "calf": {"lower": -2.4504, "upper": -0.754, "range": 1.6964}},
        },
    }


# ============================================================================
# A 类：物理自洽
# ============================================================================

def _check_force_gravity(rows: list[dict], kin: dict) -> dict:
    """检查总垂直接触力是否与体重匹配。从 URDF 可加总 link masses 得体重。

    由于 CSV 中没有直接的 mass 字段，使用 standing_height 和 leg_length 估算
    体重数量级（~12kg for Go2），做数量级检查而非精确匹配。
    """
    # 估算 base_mass 数量级：用立方律 (standing_height/0.38)^3 × 7kg（Go2 base mass）
    # 这个检查的目的是发现明显异常（如 total_force = 0 或 = 10000N）
    total_force = _col_array(rows, "total_contact_force_magnitude")
    mean_fz = float(np.mean(total_force))

    # 粗略估计：四足机器人质量 ~ standing_height^3 × 密度常数
    h = kin["standing_height"]
    estimated_mass = (h / 0.38) ** 3 * 7.0  # Go2 base = 7kg + legs
    expected_force = estimated_mass * 9.81

    ratio = mean_fz / expected_force if expected_force > 0 else 0.0
    # 容许范围 [0.5, 2.0] —— 足够宽以容纳估算误差，足够窄以排除明显异常
    return {
        "mean_total_contact_force_N": round(mean_fz, 1),
        "estimated_body_weight_N": round(expected_force, 1),
        "ratio": round(ratio, 3),
        "physically_plausible": 0.5 <= ratio <= 2.0,
        "note": "体重从 standing_height 估算，非精确值。此检查排除明显异常(ratio远偏离1.0)。",
    }


def _check_joint_limits(rows: list[dict]) -> dict:
    """检查关节是否频繁触限。直接使用 CSV 中的 joint_limit_violation_* 字段。"""
    limit_cols = [k for k in rows[0].keys() if k.startswith("joint_limit_violation_")]
    if not limit_cols:
        return {"available": False, "note": "CSV 中无 joint_limit_violation 列"}

    n_frames = len(rows)
    violations = {col: 0 for col in limit_cols}
    for row in rows:
        for col in limit_cols:
            if row.get(col, "0") in ("1", "1.0", "True"):
                violations[col] += 1

    total_violation_frames = sum(violations.values())
    return {
        "available": True,
        "total_frames": n_frames,
        "violation_frames": total_violation_frames,
        "violation_rate": round(total_violation_frames / (n_frames * len(limit_cols)), 5),
        "by_joint": {k: v for k, v in violations.items() if v > 0},
        "any_violation": total_violation_frames > 0,
    }


def _check_swing_contact_detachment(rows: list[dict]) -> dict:
    """检查摆动相期间足端接触力是否接近于零。腿在空中不应有地面反力。

    使用 current_air_time > 0 判定摆动相，net_force 的对应足分量判定接触力。
    """
    foot_labels = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    # foot_indices in net_force: FL=15, FR=16, RL=17, RR=18 (from metadata)
    foot_idx = {"FL_foot": 15, "FR_foot": 16, "RL_foot": 17, "RR_foot": 18}

    n_frames = len(rows)
    swing_frames = 0
    high_force_swing = 0

    for i, row in enumerate(rows):
        for fn in foot_labels:
            at_key = f"current_air_time_{foot_idx[fn]}"
            nf_key = f"net_force_{foot_idx[fn] * 3}"
            nf_key_1 = f"net_force_{foot_idx[fn] * 3 + 1}"
            nf_key_2 = f"net_force_{foot_idx[fn] * 3 + 2}"
            try:
                air_time = float(row.get(at_key, "0"))
                fx = float(row.get(nf_key, "0"))
                fy = float(row.get(nf_key_1, "0"))
                fz = float(row.get(nf_key_2, "0"))
                fmag = math.sqrt(fx ** 2 + fy ** 2 + fz ** 2)
                if air_time > 0:
                    swing_frames += 1
                    if fmag > 5.0:  # >5N in swing phase = foot dragging or sensor noise
                        high_force_swing += 1
            except (ValueError, KeyError):
                continue

    ratio = high_force_swing / swing_frames if swing_frames > 0 else 0.0
    return {
        "total_swing_frames": swing_frames,
        "frames_with_high_force_in_swing": high_force_swing,
        "ratio": round(ratio, 4),
        "clean_detachment": ratio < 0.02,
        "note": "摆动相中接触力>5N的帧占比。>2%表示足端拖地或传感器配置问题。",
    }


# ============================================================================
# B 类：模式自洽（时序特征）
# ============================================================================

def _detect_gait_cycles(rows: list[dict]) -> dict:
    """从接触传感器检测步态周期。返回每足周期时长序列和离群分析。"""
    foot_labels = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    foot_idx = {"FL_foot": 15, "FR_foot": 16, "RL_foot": 17, "RR_foot": 18}

    n_frames = len(rows)
    cycles = {fn: [] for fn in foot_labels}
    stance_ratios = {fn: [] for fn in foot_labels}

    for fn in foot_labels:
        idx = foot_idx[fn]
        ct_key = f"current_contact_time_{idx}"
        at_key = f"current_air_time_{idx}"

        contact = [float(row.get(ct_key, "0")) > 0 for row in rows]
        air = [float(row.get(at_key, "0")) > 0 for row in rows]

        # 着地事件（上升沿）
        touchdowns = [f for f in range(1, n_frames)
                      if contact[f] and not contact[f - 1]]
        liftoffs = [f for f in range(1, n_frames)
                    if not contact[f] and contact[f - 1]]

        # 周期 = 相邻着地事件间隔
        for i in range(1, len(touchdowns)):
            cycles[fn].append(touchdowns[i] - touchdowns[i])

        # 支撑相占比
        td_idx, lo_idx = 0, 0
        while td_idx < len(touchdowns) - 1 and lo_idx < len(liftoffs):
            if liftoffs[lo_idx] > touchdowns[td_idx] and liftoffs[lo_idx] < touchdowns[td_idx + 1]:
                dur = touchdowns[td_idx + 1] - touchdowns[td_idx]
                if dur > 0:
                    stance_ratios[fn].append((liftoffs[lo_idx] - touchdowns[td_idx]) / dur)
                td_idx += 1
                lo_idx += 1
            elif liftoffs[lo_idx] <= touchdowns[td_idx]:
                lo_idx += 1
            else:
                td_idx += 1

    # 统计每足周期，IQR 离群检测
    result = {"per_foot": {}}
    all_cycles = []
    for fn in foot_labels:
        cyc = cycles[fn]
        all_cycles.extend(cyc)
        if len(cyc) < 3:
            result["per_foot"][fn] = {"n_cycles": len(cyc), "note": "周期不足3个，无法分析"}
            continue

        arr = np.array(cyc, dtype=float)
        q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = [(i, round(v, 1)) for i, v in enumerate(cyc) if v < lower or v > upper]

        result["per_foot"][fn] = {
            "n_cycles": len(cyc),
            "mean_frames": round(float(np.mean(arr)), 1),
            "std_frames": round(float(np.std(arr)), 1),
            "cv": round(float(np.std(arr)) / float(np.mean(arr)), 3) if np.mean(arr) > 0 else None,
            "iqr_range": (round(lower, 1), round(upper, 1)),
            "outliers": [{"index": o[0], "value_frames": o[1]} for o in outliers],
            "has_outliers": len(outliers) > 0,
        }

    # 全部周期的 CV
    if all_cycles:
        all_arr = np.array(all_cycles, dtype=float)
        result["overall"] = {
            "n_total_cycles": len(all_cycles),
            "mean_frames": round(float(np.mean(all_arr)), 1),
            "std_frames": round(float(np.std(all_arr)), 1),
            "cv": round(float(np.std(all_arr)) / float(np.mean(all_arr)), 3) if np.mean(all_arr) > 0 else None,
        }

    return result


def _fft_periodicity(rows: list[dict], signal_cols: list[str], fs: float = 100.0) -> dict:
    """对指定信号做 FFT，提取主频、频谱集中度、信噪比。"""
    n = len(rows)
    if n < 64:
        return {"error": f"数据不足64帧({n}帧)，无法做频谱分析"}

    results = {}
    for col in signal_cols:
        try:
            signal = _col_array(rows, col)
            signal = signal - np.mean(signal)  # 去直流

            # FFT
            fft = np.fft.rfft(signal)
            freqs = np.fft.rfftfreq(n, d=1.0 / fs)
            mag = np.abs(fft)

            # 排除 DC
            non_dc = slice(1, None)
            freqs = freqs[non_dc]
            mag = mag[non_dc]

            if len(mag) < 3:
                continue

            # 主频
            peak_idx = int(np.argmax(mag))
            peak_freq = float(freqs[peak_idx])
            peak_mag = float(mag[peak_idx])

            # 频谱集中度：主峰周围 ±0.5Hz 的能量占比
            half_width = int(0.5 / (fs / n)) + 1
            peak_band = slice(max(0, peak_idx - half_width), min(len(mag), peak_idx + half_width + 1))
            band_energy = float(np.sum(mag[peak_band] ** 2))
            total_energy = float(np.sum(mag ** 2))
            concentration = band_energy / total_energy if total_energy > 0 else 0.0

            # 次高峰（周期性的另一证据）
            if len(mag) > 3:
                sorted_idx = np.argsort(mag)[::-1]
                second_idx = sorted_idx[1]
                second_freq = float(freqs[second_idx])
                second_ratio = float(mag[second_idx] / peak_mag) if peak_mag > 0 else 0.0
            else:
                second_freq = 0.0
                second_ratio = 0.0

            results[col] = {
                "dominant_freq_Hz": round(peak_freq, 3),
                "dominant_magnitude": round(peak_mag, 1),
                "spectral_concentration": round(concentration, 3),
                "second_peak_ratio": round(second_ratio, 3),
                "periodic_clear": concentration > 0.4 and second_ratio < 0.5,
            }

        except Exception:
            continue

    return results


def _cross_correlation_phase(rows: list[dict], pairs: list[tuple[str, str]], max_lag: int = 50) -> dict:
    """计算信号对的互相关，提取相位关系（领先-滞后）。"""
    n = len(rows)
    if n < max_lag * 2:
        max_lag = n // 4

    results = {}
    for a_name, b_name in pairs:
        try:
            a = _col_array(rows, a_name)
            b = _col_array(rows, b_name)
            a = a - np.mean(a)
            b = b - np.mean(b)

            # 互相关
            xcorr = np.correlate(a, b, mode='full')
            lags = np.arange(-n + 1, n)
            # 取最大 lag 范围内
            mask = np.abs(lags) <= max_lag
            valid_xcorr = xcorr[mask]
            valid_lags = lags[mask]

            best_idx = int(np.argmax(np.abs(valid_xcorr)))
            best_lag = int(valid_lags[best_idx])
            best_corr = float(valid_xcorr[best_idx]) / (np.std(a) * np.std(b) * n) if np.std(a) * np.std(b) > 0 else 0.0

            # 零 lag 相关
            zero_idx = int(np.where(lags == 0)[0][0])
            zero_corr = float(xcorr[zero_idx]) / (np.std(a) * np.std(b) * n) if np.std(a) * np.std(b) > 0 else 0.0

            results[f"{a_name}<->{b_name}"] = {
                "best_lag_frames": best_lag,
                "best_cross_corr": round(best_corr, 3),
                "zero_lag_corr": round(zero_corr, 3),
            }
        except Exception:
            continue

    return results


def _front_half_vs_back_half(rows: list[dict], metrics: list[str]) -> dict:
    """前后半段 Mann-Whitney U 检验。报告显著变化的指标。"""
    n = len(rows)
    if n < 100:
        return {"error": f"数据不足100帧({n}帧)"}

    mid = n // 2
    from scipy.stats import mannwhitneyu

    significant = []
    all_results = {}

    for col in metrics:
        try:
            first = np.array([float(r[col]) for r in rows[:mid]])
            second = np.array([float(r[col]) for r in rows[mid:]])
            u_stat, p_val = mannwhitneyu(first, second, alternative="two-sided")
            all_results[col] = {
                "first_half_mean": round(float(np.mean(first)), 4),
                "second_half_mean": round(float(np.mean(second)), 4),
                "p_value": round(float(p_val), 4),
            }
            if p_val < 0.01:
                significant.append(col)
        except Exception:
            continue

    return {
        "test": "Mann-Whitney U",
        "significance_level": 0.01,
        "significant_metrics": significant,
        "all_metrics": all_results,
    }


def _left_right_symmetry(rows: list[dict]) -> dict:
    """检查左右对应关节位置差异。报告偏置量和趋势。"""
    pairs = [
        ("FL_hip", "FR_hip", 28, 29),
        ("RL_hip", "RR_hip", 30, 31),
        ("FL_thigh", "FR_thigh", 32, 33),
        ("RL_thigh", "RR_thigh", 34, 35),
        ("FL_calf", "FR_calf", 36, 37),
        ("RL_calf", "RR_calf", 38, 39),
    ]

    results = {}
    n = len(rows)
    for left_name, right_name, l_idx, r_idx in pairs:
        try:
            left = np.array([float(rows[i][f"joint_pos_{l_idx}"]) for i in range(n)])
            right = np.array([float(rows[i][f"joint_pos_{r_idx}"]) for i in range(n)])
            diff = left - right
            mean_bias = float(np.mean(diff))
            rms_diff = float(np.sqrt(np.mean(diff ** 2)))

            # 趋势：线性拟合
            x = np.arange(n)
            slope = float(np.polyfit(x, diff, 1)[0])

            results[f"{left_name}_vs_{right_name}"] = {
                "mean_bias_rad": round(mean_bias, 4),
                "rms_diff_rad": round(rms_diff, 4),
                "trend_rad_per_frame": round(slope, 6),
                "trend_desc": "drifting" if abs(slope * n) > 0.05 else "stable",
            }
        except Exception:
            continue

    return results


# ============================================================================
# C 类：衍生指标（纯统计量）
# ============================================================================

def _derived_metrics(rows: list[dict]) -> dict:
    """计算供标准对比的衍生指标，不做任何好/坏判断。"""
    n = len(rows)
    lin_x = _col_array(rows, "lin_x")
    lin_y = _col_array(rows, "lin_y")
    lin_z = _col_array(rows, "lin_z")
    ang_x = _col_array(rows, "ang_x")
    ang_y = _col_array(rows, "ang_y")
    ang_z = _col_array(rows, "ang_z")
    pos_z = _col_array(rows, "pos_z")
    cmd_0 = _col_array(rows, "cmd_0")
    cmd_1 = _col_array(rows, "cmd_1")
    cmd_2 = _col_array(rows, "cmd_2")
    proj_g0 = _col_array(rows, "projected_gravity_0")
    proj_g1 = _col_array(rows, "projected_gravity_1")
    proj_g2 = _col_array(rows, "projected_gravity_2")

    # 有指令运动段（排除零指令段）
    cmd_mag = np.sqrt(cmd_0 ** 2 + cmd_1 ** 2 + cmd_2 ** 2)
    moving = cmd_mag > 0.1
    stationary = cmd_mag < 0.05

    def _safe(vals, func):
        if len(vals) > 0:
            return round(float(func(vals)), 4)
        return None

    return {
        "pos_z": {
            "mean": _safe(pos_z, np.mean),
            "std": _safe(pos_z, np.std),
        },
        "body_attitude": {
            "projected_gravity_0_mean_abs": _safe(proj_g0, lambda x: np.mean(np.abs(x))),
            "projected_gravity_1_mean_abs": _safe(proj_g1, lambda x: np.mean(np.abs(x))),
            "projected_gravity_2_below_0_95_ratio": _safe(proj_g2, lambda x: np.mean(x < 0.95)),
        },
        "body_stability": {
            "lin_z_rms": _safe(lin_z, lambda x: np.sqrt(np.mean(x ** 2))),
            "ang_xy_rms": _safe(np.sqrt(ang_x ** 2 + ang_y ** 2), lambda x: np.sqrt(np.mean(x ** 2))),
        },
        "velocity_tracking_moving": {
            "lin_xy_rmse": _safe(
                np.sqrt((lin_x[moving] - cmd_0[moving]) ** 2 + (lin_y[moving] - cmd_1[moving]) ** 2),
                np.mean,
            ) if np.any(moving) else None,
            "ang_z_rmse": _safe(
                np.abs(ang_z[moving] - cmd_2[moving]), np.mean
            ) if np.any(moving) else None,
            "frames_analyzed": int(np.sum(moving)),
        },
        "zero_command_behavior": {
            "lin_xy_rms": _safe(
                np.sqrt(lin_x[stationary] ** 2 + lin_y[stationary] ** 2),
                lambda x: np.sqrt(np.mean(x ** 2)),
            ) if np.any(stationary) else None,
            "frames_analyzed": int(np.sum(stationary)),
        },
        "joint_power": _joint_power_stats(rows),
    }


def _joint_power_stats(rows: list[dict]) -> dict:
    """关节功率统计：RMSE 和峰值比。"""
    try:
        power = _cols_arrays(rows, "joint_power_", 12)
        if power.shape[1] == 0:
            return {}
        rms = float(np.sqrt(np.mean(power ** 2, axis=0)).mean())
        peak = float(np.max(np.abs(power)))
        return {"mean_rms_W": round(rms, 2), "peak_W": round(peak, 2)}
    except Exception:
        return {}


# ============================================================================
# 汇总 & 主入口
# ============================================================================

def execute(csv_path: str, kinematics_source: str = None) -> dict:
    try:
        headers, rows = _load_csv(csv_path)
        n_frames = len(rows)
        if n_frames < 50:
            return {"error": f"数据不足：仅 {n_frames} 帧（最少 50）"}

        kin = _load_kinematics(kinematics_source)

        # 估计采样频率
        times = [float(r["time"]) for r in rows]
        fs = (len(times) - 1) / (times[-1] - times[0]) if times[-1] > times[0] else 100.0

        # === A 类：物理自洽 ===
        a_results = {
            "force_gravity": _check_force_gravity(rows, kin),
            "joint_limits": _check_joint_limits(rows),
            "swing_contact_detachment": _check_swing_contact_detachment(rows),
        }

        # === B 类：模式自洽 ===
        gait = _detect_gait_cycles(rows)

        # FFT on key signals
        fft_signals = ["pos_z", "lin_x", "lin_y", "lin_z",
                       "joint_pos_4", "joint_pos_8"]  # FL_thigh, FL_calf
        available_signals = [s for s in fft_signals if s in headers]
        fft = _fft_periodicity(rows, available_signals, fs)

        # 互相关：对角线足端接触
        xcorr_pairs = [
            ("current_air_time_15", "current_air_time_17"),  # FL vs RL
            ("current_air_time_16", "current_air_time_18"),  # FR vs RR
            ("current_air_time_15", "current_air_time_18"),  # FL vs RR (diagonal)
            ("current_air_time_16", "current_air_time_17"),  # FR vs RL (diagonal)
        ]
        available_xcorr = [(a, b) for a, b in xcorr_pairs if a in headers and b in headers]
        xcorr_results = _cross_correlation_phase(rows, available_xcorr)

        # 左右对称（原始关节位置差值）
        lr_symmetry = _left_right_symmetry(rows)

        # 前后半段对比
        drift_metrics = ["pos_z", "lin_x", "lin_y", "lin_z", "projected_gravity_2"]
        drift_metrics = [m for m in drift_metrics if m in headers]
        drift = _front_half_vs_back_half(rows, drift_metrics)

        b_results = {
            "gait_cycles": gait,
            "fft_periodicity": fft,
            "cross_correlation": xcorr_results,
            "left_right_symmetry": lr_symmetry,
            "front_half_vs_back_half": drift,
        }

        # === C 类：衍生指标 ===
        c_results = _derived_metrics(rows)

        # === 汇总 ===
        a_failures = [k for k, v in a_results.items()
                      if isinstance(v, dict) and v.get("any_violation") is True
                      or isinstance(v, dict) and "physically_plausible" in v and not v["physically_plausible"]
                      or isinstance(v, dict) and "clean_detachment" in v and not v["clean_detachment"]]

        summary = {
            "n_frames": n_frames,
            "estimated_fps": round(fs, 0),
            "physical_consistency": {
                "all_pass": len(a_failures) == 0,
                "failures": a_failures,
            },
            "pattern_highlights": _summarize_patterns(b_results),
        }

        return {
            "summary": summary,
            "physical_consistency": a_results,
            "pattern_self_consistency": b_results,
            "derived_metrics": c_results,
        }

    except Exception as e:
        return {"error": f"behavior_selfcheck failed: {type(e).__name__}: {e}"}


def _summarize_patterns(b: dict) -> list[str]:
    """从 B 类结果中提取关键发现，用自然语言描述。"""
    highlights = []

    # 步态
    gait = b.get("gait_cycles", {})
    overall = gait.get("overall", {})
    if overall:
        cv = overall.get("cv")
        if cv is not None:
            highlights.append(f"步态周期变异系数 CV={cv:.2f}（{overall['n_total_cycles']}个周期，均值{overall['mean_frames']:.0f}帧）")
        for fn, info in gait.get("per_foot", {}).items():
            if info.get("has_outliers"):
                highlights.append(f"{fn}: {len(info['outliers'])}个离群周期")

    # FFT
    fft = b.get("fft_periodicity", {})
    for col, info in fft.items():
        if isinstance(info, dict) and "dominant_freq_Hz" in info:
            if info.get("periodic_clear"):
                highlights.append(f"{col}: 主频{info['dominant_freq_Hz']}Hz，频谱集中度{info['spectral_concentration']:.0%}")
            elif info.get("spectral_concentration", 0) < 0.3:
                highlights.append(f"{col}: 频谱分散（集中度{info['spectral_concentration']:.0%}），缺乏主导周期")

    # 前后漂移
    drift = b.get("front_half_vs_back_half", {})
    sig_metrics = drift.get("significant_metrics", [])
    if sig_metrics:
        highlights.append(f"前后半段显著变化: {', '.join(sig_metrics)}")

    # 左右不对称
    lr = b.get("left_right_symmetry", {})
    for pair_name, info in lr.items():
        if isinstance(info, dict):
            if info.get("trend_desc") == "drifting":
                highlights.append(f"{pair_name}: 左右差异持续漂移")
            elif abs(info.get("mean_bias_rad", 0)) > 0.15:
                highlights.append(f"{pair_name}: 均值偏置{info['mean_bias_rad']:.3f}rad")

    return highlights
