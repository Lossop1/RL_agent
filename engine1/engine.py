"""
运动学特征解析器 v15
输入: my_play.py 采集的 CSV
输出: 纯文本特征报告
"""

import numpy as np
import argparse
import os


# ============================================================
# 配置
# ============================================================

REQUIRED_COLUMNS = [
    "time",
    "pos_z", "quat_w", "quat_x", "quat_y", "quat_z",
    "lin_x", "lin_y", "lin_z",
    "ang_x", "ang_y", "ang_z",
    "foot_FL_foot_pos_x", "foot_FR_foot_pos_x",
    "foot_RL_foot_pos_x", "foot_RR_foot_pos_x",
    "foot_height_FL_foot", "foot_height_FR_foot",
    "foot_height_RL_foot", "foot_height_RR_foot",
] + [f"joint_pos_{i}" for i in range(12)]

AC_N_POINTS = 101
CC_LAG_RANGE = 5.0

JOINT_NAMES = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]

AC_VARS = [
    "pos_z", "roll", "pitch",
    "lin_x", "lin_y", "ang_z",
    "FL_foot_z", "FR_foot_z", "RL_foot_z", "RR_foot_z",
]

CC_PAIRS = [
    ("FL_foot_z", "FR_foot_z"),
    ("FL_foot_z", "RL_foot_z"),
    ("FL_foot_z", "RR_foot_z"),
    ("FR_foot_z", "RL_foot_z"),
    ("FR_foot_z", "RR_foot_z"),
    ("RL_foot_z", "RR_foot_z"),
]

A_VARS_BODY = [
    "pos_z", "roll", "pitch",
    "lin_x", "lin_y", "lin_z",
    "ang_x", "ang_y", "ang_z",
]
A_VARS_FOOT_X = ["FL_foot_x", "FR_foot_x", "RL_foot_x", "RR_foot_x"]
A_VARS_FOOT_Z = ["FL_foot_z", "FR_foot_z", "RL_foot_z", "RR_foot_z"]
A_VARS_JOINT = [f"joint_{i}" for i in range(12)]


# ============================================================
# 数据加载
# ============================================================

def _load_csv(path: str) -> dict[str, np.ndarray]:
    with open(path, 'r') as f:
        import csv
        reader = csv.reader(f)
        header = next(reader)
    
    col_idx = {name: header.index(name) for name in REQUIRED_COLUMNS if name in header}
    missing = set(REQUIRED_COLUMNS) - set(col_idx.keys())
    if missing:
        raise ValueError(f"CSV 缺少列: {missing}")
    
    data = {name: [] for name in REQUIRED_COLUMNS}
    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            for name in REQUIRED_COLUMNS:
                data[name].append(float(row[col_idx[name]]))
    
    return {name: np.array(data[name]) for name in REQUIRED_COLUMNS}


# ============================================================
# 四元数 -> 欧拉角
# ============================================================

def _quat_to_euler(qw, qx, qy, qz):
    sinr = 2.0 * (qw * qx + qy * qz)
    cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr, cosr)

    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = np.where(np.abs(sinp) >= 1.0,
                     np.sign(sinp) * np.pi / 2,
                     np.arcsin(sinp))

    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny, cosy)

    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


# ============================================================
# 统计工具
# ============================================================

def _fivenum(x: np.ndarray) -> dict:
    q = np.quantile(x, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "min": q[0], "Q1": q[1], "median": q[2],
        "Q3": q[3], "max": q[4],
        "mean": np.mean(x), "std": np.std(x),
    }

def _autocorr(x: np.ndarray, n_points: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
    x = x - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, n=2*n)
    ac_full = np.fft.irfft(f * np.conj(f))[:n]
    ac_full = ac_full / ac_full[0]
    
    max_lag = min(n - 1, int(n_points * 1.5))
    lags = np.arange(max_lag) * dt
    ac = ac_full[:max_lag]
    
    idx = np.linspace(0, max_lag - 1, n_points).astype(int)
    return lags[idx], ac[idx]

def _crosscorr(x: np.ndarray, y: np.ndarray, lag_range: float,
               n_points: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
    x = x - np.mean(x)
    y = y - np.mean(y)
    n = len(x)
    
    f_x = np.fft.rfft(x, n=2*n)
    f_y = np.fft.rfft(y, n=2*n)
    cc_full = np.fft.irfft(f_x * np.conj(f_y))[:n]
    denom = np.sqrt(np.sum(x**2) * np.sum(y**2))
    if denom > 0:
        cc_full = cc_full / denom
    
    max_lag_samples = int(lag_range / dt)
    cc_neg = cc_full[-1:-max_lag_samples-1:-1][::-1]
    cc_pos = cc_full[:max_lag_samples + 1]
    cc = np.concatenate([cc_neg, cc_pos])
    lags = np.arange(-max_lag_samples, max_lag_samples + 1) * dt
    
    idx = np.linspace(0, len(lags) - 1, n_points).astype(int)
    return lags[idx], cc[idx]

def _histogram(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts, edges = np.histogram(x, bins='fd')
    return edges, counts


# ============================================================
# 数据预处理
# ============================================================

def _build_dataframe(data: dict) -> dict[str, np.ndarray]:
    df = {}
    
    df["pos_z"] = data["pos_z"]
    roll, pitch, yaw = _quat_to_euler(
        data["quat_w"], data["quat_x"], data["quat_y"], data["quat_z"]
    )
    df["roll"] = roll
    df["pitch"] = pitch
    df["yaw"] = yaw
    df["lin_x"] = data["lin_x"]
    df["lin_y"] = data["lin_y"]
    df["lin_z"] = data["lin_z"]
    df["ang_x"] = data["ang_x"]
    df["ang_y"] = data["ang_y"]
    df["ang_z"] = data["ang_z"]
    
    df["FL_foot_x"] = data["foot_FL_foot_pos_x"]
    df["FR_foot_x"] = data["foot_FR_foot_pos_x"]
    df["RL_foot_x"] = data["foot_RL_foot_pos_x"]
    df["RR_foot_x"] = data["foot_RR_foot_pos_x"]
    df["FL_foot_z"] = data["foot_height_FL_foot"]
    df["FR_foot_z"] = data["foot_height_FR_foot"]
    df["RL_foot_z"] = data["foot_height_RL_foot"]
    df["RR_foot_z"] = data["foot_height_RR_foot"]
    
    for i in range(12):
        df[f"joint_{i}"] = data[f"joint_pos_{i}"]
    
    return df


# ============================================================
# 各节格式化
# ============================================================

def _make_head(data: dict) -> str:
    dt = float(np.median(np.diff(data["time"])))
    n_frames = len(data["time"])
    duration = n_frames * dt
    joint_mapping = ", ".join([f"joint_{i}={JOINT_NAMES[i]}" for i in range(12)])
    
    return "\n".join([
        "# ============================================================",
        "# 运动学特征报告",
        f"# 帧数: {n_frames}  dt: {dt:.3f}s  时长: {duration:.1f}s",
        f"# 关节映射: {joint_mapping}",
        "# ============================================================",
    ])


def _compute_section_A(df: dict) -> str:
    lines = [
        "# ----------------------------------------------------------",
        "# A. 时域统计",
        "# 格式: name min Q1 median Q3 max mean std (空格分隔)",
        "# 字段: 1=name 2=min 3=Q1 4=median 5=Q3 6=max 7=mean 8=std",
        "# min=最小值 Q1=第一四分位数 median=中位数 Q3=第三四分位数",
        "# max=最大值 mean=均值 std=标准差",
        "# 单位: m(位置) deg(角度) m/s(线速度) deg/s(角速度)",
        "# ----------------------------------------------------------",
    ]
    
    for name in A_VARS_BODY + A_VARS_FOOT_X + A_VARS_FOOT_Z + A_VARS_JOINT:
        s = _fivenum(df[name])
        lines.append(
            f"{name} {s['min']:.4f} {s['Q1']:.4f} {s['median']:.4f} "
            f"{s['Q3']:.4f} {s['max']:.4f} {s['mean']:.4f} {s['std']:.4f}"
        )
    
    return "\n".join(lines)


def _compute_section_B(df: dict, dt: float) -> str:
    lines = [
        "# ----------------------------------------------------------",
        "# B. 自相关",
        "# 格式: ## 变量名, 下一行开始 lag value (空格分隔, 101 点)",
        "# ----------------------------------------------------------",
    ]
    
    for name in AC_VARS:
        lags, ac = _autocorr(df[name], AC_N_POINTS, dt)
        lines.append(f"## {name}")
        for lag, val in zip(lags, ac):
            lines.append(f"{lag:.3f} {val:.6f}")
    
    return "\n".join(lines)


def _compute_section_C(df: dict, dt: float) -> str:
    lines = [
        "# ----------------------------------------------------------",
        "# C. 互相关 (foot_z)",
        "# 6 对: FL-FR, FL-RL, FL-RR, FR-RL, FR-RR, RL-RR",
        "# 格式: ## pair_name, 下一行开始 lag value (空格分隔, 101 点)",
        "# ----------------------------------------------------------",
    ]
    
    for a, b in CC_PAIRS:
        lags, cc = _crosscorr(df[a], df[b], CC_LAG_RANGE, AC_N_POINTS, dt)
        pair_name = f"{a} x {b}"
        lines.append(f"## {pair_name}")
        for lag, val in zip(lags, cc):
            lines.append(f"{lag:.3f} {val:.6f}")
    
    return "\n".join(lines)


def _compute_section_D(df: dict) -> str:
    pos_z = df["pos_z"]
    pos_z_med = np.median(pos_z)
    mask = np.abs(pos_z - pos_z_med) < 0.01
    nominal = {}
    for leg in A_VARS_FOOT_Z:
        nominal[leg] = np.median(df[leg][mask]) if np.any(mask) else np.median(df[leg])
    
    lines = [
        "# ----------------------------------------------------------",
        "# D. 足端高度分布",
        "# 格式: ## leg_name, 下一行开始 [left, right): count (空格分隔)",
        "# bins: [left, right)",
        f"# nominal: FL={nominal['FL_foot_z']:.3f} FR={nominal['FR_foot_z']:.3f} "
        f"RL={nominal['RL_foot_z']:.3f} RR={nominal['RR_foot_z']:.3f}",
        "# ----------------------------------------------------------",
    ]
    
    for leg in A_VARS_FOOT_Z:
        edges, counts = _histogram(df[leg])
        lines.append(f"## {leg}")
        for i in range(len(counts)):
            lines.append(f"[{edges[i]:.3f}, {edges[i+1]:.3f}): {counts[i]}")
    
    return "\n".join(lines)


def _compute_section_E(df: dict) -> str:
    lines = [
        "# ----------------------------------------------------------",
        "# E. 关节角度分布",
        "# 格式: ## joint_N, 下一行开始 [left, right): count (空格分隔)",
        "# bins: [left, right)",
        "# ----------------------------------------------------------",
    ]
    
    for i in range(12):
        edges, counts = _histogram(df[f"joint_{i}"])
        lines.append(f"## joint_{i}")
        for j in range(len(counts)):
            lines.append(f"[{edges[j]:.3f}, {edges[j+1]:.3f}): {counts[j]}")
    
    return "\n".join(lines)


# ============================================================
# 主入口
# ============================================================

def parse(csv_path: str) -> str:
    raw = _load_csv(csv_path)
    df = _build_dataframe(raw)
    dt = float(np.median(np.diff(raw["time"])))
    
    sections = [
        _make_head(raw),
        _compute_section_A(df),
        _compute_section_B(df, dt),
        _compute_section_C(df, dt),
        _compute_section_D(df),
        _compute_section_E(df),
    ]
    
    return "\n\n".join(sections)


def main():
    parser = argparse.ArgumentParser(description="运动学特征解析器 v15")
    parser.add_argument("csv", help="my_play.py 采集的 CSV 文件路径")
    parser.add_argument("-o", "--output", default=None,
                        help="输出文件路径 (默认: 与 CSV 同名 _features.txt)")
    args = parser.parse_args()
    
    output_path = args.output or os.path.splitext(args.csv)[0] + "_features.txt"
    report = parse(args.csv)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"特征报告已保存至: {output_path}")


if __name__ == "__main__":
    main()