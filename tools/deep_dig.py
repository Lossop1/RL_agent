"""
deep_dig.py — 训练数据深度挖掘工具（修正版 v2.0）

从 parse_log 产出的完整 JSON 中提取时序模式：
- 变点检测：趋势结构性变化的位置
- 事件关联：指标间的时间延迟和因果线索
- 异常检测：单轮或短区间的异常模式
- 趋势预测：未来趋势的外推和收敛估计
- 效果量化：调参前后跨快照的统计检验

修正记录：
v2.0 - 统计推断严谨性重构
  - PELT：修正为正确剪枝实现 + O(1) cost 计算
  - 格兰杰：修正为 F 分布生存函数 + 多重比较 FDR 校正 + 数值诊断
  - 孤立森林：修正为正确的逐样本深度追踪（建议生产环境使用 sklearn）
  - 统一诊断契约：所有函数返回结构化状态 + 原因 + 数据
  - 防御性编程：源头数值检查、条件数警告、静默失败消除

MCP Tool: deep_dig
"""
import json
import os
import sys
import argparse
import numpy as np
from pathlib import Path
from enum import Enum
from typing import Optional, Union, Any
from scipy.stats import mannwhitneyu, f as f_dist
from scipy.optimize import curve_fit
from statsmodels.stats.multitest import multipletests

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
# 诊断状态系统
# ============================================================

class AnalysisStatus(Enum):
    """分析状态：可操作性级别"""
    OK = "ok"
    WARNING = "warning"        # 结果可用，但存在需要注意的异常
    ERROR = "error"            # 计算失败，结果不可用
    UNAVAILABLE = "unavailable"  # 数据不足，无法执行分析

class AnalysisReason(Enum):
    """分析失败/警告的具体原因"""
    SUCCESS = "success"
    INSUFFICIENT_DATA = "insufficient_data"
    SERIES_TOO_SHORT = "series_too_short"
    NO_VALID_PAIRS = "no_valid_pairs"
    HIGH_CONDITION_NUMBER = "high_condition_number"
    NEAR_PERFECT_FIT = "near_perfect_fit"
    EXTREME_F_STATISTIC = "extreme_f_statistic"
    NEGATIVE_COST_CLAMPED = "negative_cost_clamped"
    NON_CONVERGING_SERIES = "non_converging_series"
    TREND_EXPLOSION = "trend_explosion"
    VALUE_ERROR = "value_error"
    LIN_ALG_ERROR = "lin_alg_error"
    UNEXPECTED_ERROR = "unexpected_error"
    MISSING_BUFFER_DATA = "missing_buffer_data"
    PREVIOUS_SNAPSHOT_NOT_FOUND = "previous_snapshot_not_found"
    FEATURE_EXTRACTION_FAILED = "feature_extraction_failed"

class AnalysisResult:
    """统一的诊断分析结果封装"""
    def __init__(self, status: AnalysisStatus, reason: AnalysisReason, 
                 data: Any = None, detail: Optional[str] = None,
                 warnings: Optional[list] = None):
        self.status = status
        self.reason = reason
        self.data = data
        self.detail = detail
        self.warnings = warnings or []
    
    def to_dict(self):
        result = {
            "status": self.status.value,
            "reason": self.reason.value,
        }
        if self.data is not None:
            result["data"] = self.data
        if self.detail is not None:
            result["detail"] = self.detail
        if self.warnings:
            result["warnings"] = self.warnings
        return result
    
    @classmethod
    def ok(cls, data: Any, warnings: Optional[list] = None):
        return cls(AnalysisStatus.OK, AnalysisReason.SUCCESS, 
                   data=data, warnings=warnings)
    
    @classmethod
    def warning(cls, reason: AnalysisReason, data: Any = None, 
                detail: Optional[str] = None, warnings: Optional[list] = None):
        return cls(AnalysisStatus.WARNING, reason, 
                   data=data, detail=detail, warnings=warnings)
    
    @classmethod
    def error(cls, reason: AnalysisReason, detail: Optional[str] = None):
        return cls(AnalysisStatus.ERROR, reason, 
                   data=None, detail=detail)
    
    @classmethod
    def unavailable(cls, reason: AnalysisReason, detail: Optional[str] = None):
        return cls(AnalysisStatus.UNAVAILABLE, reason, 
                   data=None, detail=detail)

# ============================================================
# JSON 清洗
# ============================================================

def _clean_for_json(obj: Any) -> Any:
    """递归清洗 numpy 类型，确保 JSON 可序列化"""
    if isinstance(obj, dict):
        return {str(k): _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, AnalysisResult):
        return obj.to_dict()
    if isinstance(obj, (AnalysisStatus, AnalysisReason)):
        return obj.value
    if isinstance(obj, (np.generic,)):
        return obj.item()
    return obj

# ============================================================
# 辅助函数
# ============================================================

def _load_snapshot(path: str) -> dict:
    """加载快照 JSON 文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _find_previous_snapshot(data_dir: str, current_path: str) -> Optional[dict]:
    """
    在快照目录中找当前快照之前的最新快照。
    使用时间戳精确比较，避免字符串偶发正确性。
    """
    from datetime import datetime
    
    if not os.path.isdir(data_dir):
        return None
    
    # 收集所有快照及其时间戳
    all_snapshots = []
    for sub in os.listdir(data_dir):
        sub_path = os.path.join(data_dir, sub)
        if not os.path.isdir(sub_path):
            continue
        
        # 验证时间戳格式
        try:
            ts = datetime.strptime(sub, '%Y-%m-%d_%H-%M-%S')
        except ValueError:
            continue
        
        for fname in os.listdir(sub_path):
            if fname.startswith("iter") and fname.endswith(".json"):
                all_snapshots.append({
                    "path": os.path.join(sub_path, fname),
                    "timestamp": ts,
                    "filename": fname
                })
    
    if not all_snapshots:
        return None
    
    # 解析当前快照的时间戳
    current_ts_str = os.path.basename(os.path.dirname(current_path))
    try:
        current_dt = datetime.strptime(current_ts_str, '%Y-%m-%d_%H-%M-%S')
    except ValueError:
        return None
    
    # 找到前一个快照
    previous = [
        s for s in all_snapshots 
        if s["timestamp"] < current_dt or 
        (s["timestamp"] == current_dt and s["filename"] < os.path.basename(current_path))
    ]
    
    if not previous:
        return None
    
    previous.sort(key=lambda s: (s["timestamp"], s["filename"]), reverse=True)
    try:
        return _load_snapshot(previous[0]["path"])
    except Exception:
        return None

# ============================================================
# 一、变点检测 (PELT - 正确实现)
# ============================================================

def _pelt_cost(cumsum: np.ndarray, cumsum_sq: np.ndarray, s: int, t: int) -> float:
    """
    计算 arr[s:t] 的平方损失 cost = sum((x - mean)²)
    使用累积和实现 O(1) 计算，带浮点误差防御。
    """
    if s == t:
        return 0.0
    
    length = t - s
    sum_val = cumsum[t] - cumsum[s]
    sum_sq = cumsum_sq[t] - cumsum_sq[s]
    
    raw_cost = sum_sq - (sum_val ** 2) / length
    
    if raw_cost < 0:
        if raw_cost > -1e-10:
            return 0.0  # 浮点噪声，安全 clamp
        else:
            raise RuntimeError(
                f"Negative cost {raw_cost:.6e} at s={s}, t={t}. "
                f"sum_sq={sum_sq:.6f}, sum_val={sum_val:.6f}, length={length}. "
                f"Possible data corruption in cumulative sums."
            )
    return raw_cost


def detect_change_points(series: list, pen: float = 10.0) -> list[int]:
    """
    PELT (Pruned Exact Linear Time) 算法检测均值变点。
    
    使用标准 PELT 剪枝条件和累积和技巧实现 O(1) cost 计算。
    平均时间复杂度 O(n)，最坏情况 O(n²)。
    
    Args:
        series: 时间序列数据
        pen: 变点惩罚项（越大越不容易检测到变点）
    
    Returns:
        变点位置列表（每个变点后的第一个索引）
    """
    n = len(series)
    if n < 20:
        return []
    
    arr = np.array(series, dtype=float)
    
    # 累积和技巧
    cumsum = np.zeros(n + 1)
    cumsum[1:] = np.cumsum(arr)
    cumsum_sq = np.zeros(n + 1)
    cumsum_sq[1:] = np.cumsum(arr ** 2)
    
    # DP 数组
    F = np.full(n + 1, np.inf)
    F[0] = -pen  # 允许序列从位置0开始
    cp = np.zeros(n + 1, dtype=int)
    candidates = [0]  # PELT 候选集
    
    for t in range(1, n + 1):
        best_cost = np.inf
        best_s = 0
        
        # 只在候选集中搜索
        for s in candidates:
            total_cost = F[s] + _pelt_cost(cumsum, cumsum_sq, s, t) + pen
            if total_cost < best_cost:
                best_cost = total_cost
                best_s = s
        
        F[t] = best_cost
        cp[t] = best_s
        
        # PELT 剪枝条件：保留满足 F[s] + cost(s,t) + pen <= F[t] 的候选点
        # 等价于：若 F[s] + cost(s,t) + pen > F[t]，则 s 永远不可能是未来最优前驱
        new_candidates = []
        for s in candidates:
            if F[s] + _pelt_cost(cumsum, cumsum_sq, s, t) + pen <= F[t]:
                new_candidates.append(s)
        new_candidates.append(t)
        candidates = new_candidates
    
    # 回溯变点
    change_points = []
    t = n
    while t > 0:
        s = cp[t]
        if s > 0:
            change_points.append(s)
        t = s
    
    return sorted(change_points)


def compute_all_change_points(buffer_data: dict, 
                              trend_fields: list) -> AnalysisResult:
    """对所有 trend_fields 做变点检测"""
    results = []
    warnings = []
    errors = []
    
    for field in trend_fields:
        try:
            series = [snap.get(field) for snap in buffer_data.values()]
            series = [v for v in series if v is not None]
            
            if len(series) < 20:
                errors.append({
                    "field": field,
                    "reason": AnalysisReason.SERIES_TOO_SHORT.value,
                    "detail": f"series length {len(series)} < 20"
                })
                continue
            
            cps = detect_change_points(series)
            
            for cp_idx in cps:
                before = np.mean(series[:cp_idx])
                after = np.mean(series[cp_idx:])
                
                if abs(before) > 1e-8:
                    pct = (after - before) / abs(before) * 100
                else:
                    pct = float("inf") if after != 0 else 0
                
                results.append({
                    "metric": field,
                    "iteration": int(cp_idx),
                    "before_mean": round(float(before), 4),
                    "after_mean": round(float(after), 4),
                    "change_pct": round(float(pct), 1),
                })
        except Exception as e:
            errors.append({
                "field": field,
                "reason": AnalysisReason.UNEXPECTED_ERROR.value,
                "detail": str(e)
            })
    
    if not results and errors:
        return AnalysisResult.error(
            AnalysisReason.FEATURE_EXTRACTION_FAILED,
            f"All {len(trend_fields)} fields failed: " + 
            "; ".join([e["detail"] for e in errors[:3]])
        )
    
    if errors:
        warnings.append({
            "type": "partial_failures",
            "detail": f"{len(errors)} fields failed to process",
            "failed_fields": [e["field"] for e in errors]
        })
    
    return AnalysisResult.ok(results, warnings=warnings if warnings else None)

# ============================================================
# 二、事件关联 (交叉相关 + 格兰杰)
# ============================================================

def compute_cross_correlation(x: list, y: list, max_lag: int = 30) -> tuple[int, float]:
    """计算两个序列的交叉相关，返回最大相关和对应延迟"""
    n = len(x)
    best_lag, best_corr = 0, 0.0
    
    for lag in range(1, min(max_lag, n // 3) + 1):
        if lag < n:
            corr = np.corrcoef(x[:-lag], y[lag:])[0, 1]
            if not np.isnan(corr) and abs(corr) > abs(best_corr):
                best_corr, best_lag = corr, lag
    
    return best_lag, best_corr


def compute_granger_causality(x: list, y: list, 
                              max_lag: int = 5) -> AnalysisResult:
    """
    格兰杰因果检验（修正版）。
    
    修正内容：
    - 使用 scipy.stats.f 的生存函数计算正确的 p 值
    - 添加条件数检查，防止病态矩阵
    - 设置合理的 rcond 阈值
    - 返回结构化诊断信息
    """
    n = len(x)
    
    # 数据充分性检查
    if n < max_lag + 10:
        return AnalysisResult.unavailable(
            AnalysisReason.INSUFFICIENT_DATA,
            f"Need at least {max_lag + 10} samples, got {n}"
        )
    
    try:
        # 构建受限制模型（仅 y 的滞后项）
        y_target = np.array(y[max_lag:], dtype=float)
        y_lags = np.column_stack(
            [np.array(y[max_lag - i - 1:n - i - 1], dtype=float) 
             for i in range(max_lag)]
        )
        
        # 剔除 NaN
        valid = ~np.isnan(y_lags).any(axis=1) & ~np.isnan(y_target)
        if valid.sum() < max_lag + 5:
            return AnalysisResult.unavailable(
                AnalysisReason.INSUFFICIENT_DATA,
                f"Only {valid.sum()} valid samples after removing NaN"
            )
        
        X_r = y_lags[valid]
        y_t = y_target[valid]
        
        # 检查病态矩阵（共线性本身是诊断信号）
        cond_num = np.linalg.cond(X_r)
        if cond_num > 1e10:
            return AnalysisResult.warning(
                AnalysisReason.HIGH_CONDITION_NUMBER,
                data={"f_stat": 0.0, "p_val": 1.0},
                detail=f"High condition number ({cond_num:.1e}) in restricted model. "
                       f"Variables may be highly collinear."
            )
        
        # 拟合受限制模型
        beta_r = np.linalg.lstsq(X_r, y_t, rcond=1e-8)[0]
        resid_r = y_t - X_r @ beta_r
        ssr_r = np.sum(resid_r ** 2)
        
        # 构建无限制模型（y 的滞后项 + x 的滞后项）
        x_lags = np.column_stack(
            [np.array(x[max_lag - i - 1:n - i - 1], dtype=float) 
             for i in range(max_lag)]
        )[valid]
        
        X_u = np.column_stack([X_r, x_lags])
        
        cond_num_u = np.linalg.cond(X_u)
        if cond_num_u > 1e10:
            return AnalysisResult.warning(
                AnalysisReason.HIGH_CONDITION_NUMBER,
                data={"f_stat": 0.0, "p_val": 1.0},
                detail=f"High condition number ({cond_num_u:.1e}) in unrestricted model."
            )
        
        # 拟合无限制模型
        beta_u = np.linalg.lstsq(X_u, y_t, rcond=1e-8)[0]
        resid_u = y_t - X_u @ beta_u
        ssr_u = np.sum(resid_u ** 2)
        
        # 计算 F 统计量
        df1 = max_lag
        df2 = len(y_t) - 2 * max_lag
        
        if df2 <= 0:
            return AnalysisResult.unavailable(
                AnalysisReason.INSUFFICIENT_DATA,
                f"Denominator degrees of freedom <= 0"
            )
        
        if ssr_u < 1e-10:
            return AnalysisResult.warning(
                AnalysisReason.NEAR_PERFECT_FIT,
                data={"f_stat": float("inf"), "p_val": 0.0},
                detail="Near-perfect fit in unrestricted model. Possible overfitting."
            )
        
        f_stat = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
        
        # 使用 F 分布的生存函数计算正确的 p 值
        p_val = f_dist.sf(f_stat, df1, df2)
        
        # 异常大的 F 统计量警告
        if f_stat > 1000:
            return AnalysisResult.warning(
                AnalysisReason.EXTREME_F_STATISTIC,
                data={
                    "f_stat": round(float(f_stat), 4),
                    "p_val": round(float(p_val), 6),
                    "df1": df1,
                    "df2": df2
                },
                detail=f"Extremely large F-statistic ({f_stat:.1f}). "
                       f"Either very strong causality or numerical instability."
            )
        
        return AnalysisResult.ok({
            "f_stat": round(float(f_stat), 4),
            "p_val": round(float(p_val), 6),
            "df1": df1,
            "df2": df2
        })
    
    except np.linalg.LinAlgError as e:
        return AnalysisResult.error(
            AnalysisReason.LIN_ALG_ERROR,
            str(e)
        )
    except Exception as e:
        return AnalysisResult.error(
            AnalysisReason.UNEXPECTED_ERROR,
            str(e)
        )


def compute_all_event_links(buffer_data: dict, 
                            all_fields: list,
                            correction_method: str = 'fdr_bh') -> AnalysisResult:
    """
    对所有指标对做交叉相关和格兰杰检验。
    
    Args:
        buffer_data: 时序缓冲数据
        all_fields: 所有指标字段
        correction_method: 多重比较校正方法
            - 'fdr_bh': Benjamini-Hochberg FDR（默认，适合探索性分析）
            - 'bonferroni': Bonferroni 校正（适合强控制场景）
    """
    first_key = next(iter(buffer_data.keys()))
    first_snap = buffer_data[first_key]
    fields = [f for f in all_fields if f in first_snap]
    
    if len(fields) < 2:
        return AnalysisResult.unavailable(
            AnalysisReason.NO_VALID_PAIRS,
            f"Only {len(fields)} valid fields found"
        )
    
    # 收集所有检验结果
    all_results = []
    all_pvals = []
    diagnostics = []
    
    n = len(fields)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            # 提取序列
            x = [snap.get(fields[i]) for snap in buffer_data.values()]
            y = [snap.get(fields[j]) for snap in buffer_data.values()]
            x = [v for v in x if v is not None]
            y = [v for v in y if v is not None]
            min_len = min(len(x), len(y))
            x, y = x[-min_len:], y[-min_len:]
            
            if len(x) < 30:
                diagnostics.append({
                    "source": fields[i],
                    "target": fields[j],
                    "status": "skipped",
                    "reason": f"Insufficient data: {len(x)} samples"
                })
                continue
            
            # 交叉相关
            lag, corr = compute_cross_correlation(x, y)
            
            if abs(corr) < 0.6:
                continue
            
            # 格兰杰检验
            granger_result = compute_granger_causality(x, y)
            
            if granger_result.status == AnalysisStatus.ERROR:
                diagnostics.append({
                    "source": fields[i],
                    "target": fields[j],
                    "status": "error",
                    "detail": granger_result.detail
                })
                continue
            
            if granger_result.status == AnalysisStatus.UNAVAILABLE:
                diagnostics.append({
                    "source": fields[i],
                    "target": fields[j],
                    "status": "unavailable",
                    "detail": granger_result.detail
                })
                continue
            
            # 收集 p 值用于多重比较校正
            p_val = granger_result.data["p_val"]
            all_pvals.append(p_val)
            all_results.append({
                "source": fields[i],
                "target": fields[j],
                "lag": int(lag),
                "cross_correlation": round(float(corr), 4),
                "granger_f": granger_result.data["f_stat"],
                "granger_p": p_val,
                "df1": granger_result.data["df1"],
                "df2": granger_result.data["df2"],
            })
    
    if not all_results:
        return AnalysisResult.ok(
            [],
            warnings=[{
                "type": "no_significant_links",
                "detail": "No pair passed the cross-correlation threshold",
                "diagnostics": diagnostics[:10] if diagnostics else None
            }] if diagnostics else None
        )
    
    # 多重比较校正
    reject, pvals_corrected, _, _ = multipletests(
        all_pvals, alpha=0.05, method=correction_method
    )
    
    # 构建最终结果
    significant = []
    for i, (res, is_sig) in enumerate(zip(all_results, reject)):
        if is_sig:
            res["granger_p_corrected"] = round(float(pvals_corrected[i]), 4)
            res["correction_method"] = correction_method
            significant.append(res)
    
    significant.sort(key=lambda r: abs(r["cross_correlation"]), reverse=True)
    
    warnings = []
    if not significant:
        warnings.append({
            "type": "no_significant_after_correction",
            "detail": f"No pair remained significant after {correction_method} correction"
        })
    if diagnostics:
        warnings.append({
            "type": "skipped_pairs",
            "count": len(diagnostics),
            "details": diagnostics[:5]
        })
    
    return AnalysisResult.ok(significant[:20], warnings=warnings if warnings else None)

# ============================================================
# 三、异常检测 (孤立森林 - 修正版)
# ============================================================

def isolation_forest(data_matrix: np.ndarray, 
                     n_estimators: int = 100, 
                     contamination: float = 0.05) -> tuple[np.ndarray, dict]:
    """
    修正版孤立森林实现。
    
    为每个样本独立计算深度，避免追踪较大分区的错误逻辑。
    
    注意：此实现用于演示正确逻辑，生产环境建议使用 
    sklearn.ensemble.IsolationForest。
    
    Returns:
        (scores, diagnostics): 异常分数（0-1，越高越异常）和诊断信息
    """
    n_samples, n_features = data_matrix.shape
    depths = np.zeros(n_samples)
    height_limit = int(np.ceil(np.log2(max(2, min(256, n_samples)))))
    
    for tree_idx in range(n_estimators):
        sample_size = min(256, n_samples)
        sample_idx = np.random.choice(n_samples, sample_size, replace=False)
        
        # 为每个样本独立追踪深度
        for i, global_idx in enumerate(sample_idx):
            depth = 0
            
            # 获取当前树中的所有样本索引
            active_idx = np.arange(sample_size)
            local_i = i  # 目标样本在 active_idx 中的当前位置
            
            while len(active_idx) > 1 and depth < height_limit:
                # 随机选择特征
                feat = np.random.randint(n_features)
                col = data_matrix[sample_idx[active_idx], feat]
                
                min_v, max_v = col.min(), col.max()
                if min_v >= max_v:
                    break
                
                split = np.random.uniform(min_v, max_v)
                left_mask = col <= split
                
                depth += 1
                
                # 确定目标样本的去向
                if left_mask[local_i]:
                    # 去了左侧
                    active_idx = active_idx[left_mask]
                    local_i = np.where(active_idx == (sample_idx == global_idx).nonzero()[0][0] 
                                      if global_idx in sample_idx[active_idx] else -1)[0]
                    if len(local_i) > 0:
                        local_i = local_i[0]
                    else:
                        break
                else:
                    # 去了右侧
                    active_idx = active_idx[~left_mask]
                    local_i = np.where(active_idx == (sample_idx == global_idx).nonzero()[0][0] 
                                      if global_idx in sample_idx[active_idx] else -1)[0]
                    if len(local_i) > 0:
                        local_i = local_i[0]
                    else:
                        break
                
                if len(active_idx) <= 1:
                    break
            
            depths[global_idx] += depth
    
    # 计算平均深度和异常分数
    avg_depth = depths / n_estimators
    
    # 使用标准归一化常数
    sample_size_ref = min(256, n_samples)
    if sample_size_ref > 2:
        c = 2 * (np.log(sample_size_ref - 1) + 0.5772156649) - 2 * (sample_size_ref - 1) / sample_size_ref
    else:
        c = 1.0
    
    scores = 2.0 ** (-avg_depth / max(c, 1e-10))
    
    diagnostics = {
        "n_estimators": n_estimators,
        "sample_size": sample_size_ref,
        "height_limit": height_limit,
        "c_factor": round(c, 4),
        "avg_depth_range": [round(float(avg_depth.min()), 2), round(float(avg_depth.max()), 2)],
        "note": "For production use, consider sklearn.ensemble.IsolationForest"
    }
    
    return scores, diagnostics


def compute_anomalies(buffer_data: dict, all_fields: list) -> AnalysisResult:
    """对 buffer 中所有轮次做异常检测"""
    try:
        iterations = sorted(buffer_data.keys())
        first_snap = buffer_data[iterations[0]]
        fields = [f for f in all_fields if f in first_snap]
        
        if not fields:
            return AnalysisResult.unavailable(
                AnalysisReason.NO_VALID_PAIRS,
                "No valid fields found in buffer data"
            )
        
        # 构建特征矩阵
        matrix = []
        for it in iterations:
            row = [buffer_data[it].get(f, 0) for f in fields]
            matrix.append(row)
        matrix = np.array(matrix, dtype=float)
        
        if matrix.shape[0] < 10:
            return AnalysisResult.unavailable(
                AnalysisReason.INSUFFICIENT_DATA,
                f"Only {matrix.shape[0]} samples, minimum 10 required"
            )
        
        # 标准化
        mean = np.mean(matrix, axis=0)
        std = np.std(matrix, axis=0)
        std[std < 1e-8] = 1.0
        matrix_norm = (matrix - mean) / std
        
        # 运行孤立森林
        scores, forest_diag = isolation_forest(matrix_norm)
        threshold = np.percentile(scores, 95)
        
        # 构建异常列表
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
        
        # 按异常分数排序
        anomalies.sort(key=lambda a: a["anomaly_score"], reverse=True)
        
        warnings = []
        if not anomalies:
            warnings.append({
                "type": "no_anomalies_detected",
                "detail": "No iterations exceeded the 95th percentile threshold"
            })
        
        return AnalysisResult.ok(
            anomalies[:10],
            warnings=warnings if warnings else None
        )
    
    except Exception as e:
        return AnalysisResult.error(
            AnalysisReason.UNEXPECTED_ERROR,
            str(e)
        )

# ============================================================
# 四、趋势预测 (Holt-Winters + 指数衰减拟合)
# ============================================================

def holt_winters_forecast(series: list, 
                          forecast_steps: int = 50, 
                          seasonal_period: int = 10,
                          damp: bool = True) -> AnalysisResult:
    """
    Holt-Winters 加法模型（修正版）。
    
    修正内容：
    - 添加趋势阻尼（damped trend）
    - 预测区间随步数膨胀（sqrt(h) * sigma）
    
    Args:
        series: 时间序列
        forecast_steps: 预测步数
        seasonal_period: 季节周期
        damp: 是否使用阻尼趋势
    """
    series = np.array(series, dtype=float)
    n = len(series)
    
    if n < seasonal_period * 2:
        return AnalysisResult.unavailable(
            AnalysisReason.SERIES_TOO_SHORT,
            f"Need at least {seasonal_period * 2} samples, got {n}"
        )
    
    try:
        # 初始化
        level = np.mean(series[:seasonal_period])
        trend = (np.mean(series[seasonal_period:2 * seasonal_period]) -
                 np.mean(series[:seasonal_period])) / seasonal_period
        seasonal = series[:seasonal_period] - level
        
        alpha, beta, gamma = 0.3, 0.1, 0.1
        phi = 0.95 if damp else 1.0  # 阻尼系数
        
        # 拟合
        for t in range(seasonal_period, n):
            old_level = level
            level = alpha * (series[t] - seasonal[t % seasonal_period]) + \
                    (1 - alpha) * (level + phi * trend)
            trend = beta * (level - old_level) + (1 - beta) * phi * trend
            seasonal[t % seasonal_period] = gamma * (series[t] - level) + \
                                           (1 - gamma) * seasonal[t % seasonal_period]
        
        # 预测
        forecast = []
        current_level = level
        current_trend = trend
        
        for step in range(1, forecast_steps + 1):
            if damp:
                # 阻尼趋势的累积
                trend_damp = current_trend * sum(phi ** k for k in range(1, step + 1))
                f = current_level + trend_damp + seasonal[(n + step - 1) % seasonal_period]
            else:
                f = current_level + step * current_trend + seasonal[(n + step - 1) % seasonal_period]
            forecast.append(round(float(f), 4))
        
        # 计算残差标准差
        residuals = []
        for i in range(n - seasonal_period):
            t = seasonal_period + i
            if damp:
                trend_est = trend * sum(phi ** k for k in range(1, i + 2))
            else:
                trend_est = (i + 1) * trend
            fitted = level + trend_est + seasonal[t % seasonal_period]
            residuals.append(series[t] - fitted)
        
        resid_std = float(np.std(residuals))
        
        # 预测区间随步数膨胀
        lower = []
        upper = []
        for h in range(1, forecast_steps + 1):
            inflation = np.sqrt(h) * resid_std  # 简化：区间随步数线性膨胀
            lower.append(round(forecast[h-1] - 1.96 * inflation, 4))
            upper.append(round(forecast[h-1] + 1.96 * inflation, 4))
        
        # 检查趋势是否爆炸
        warnings = []
        if damp and abs(current_trend * phi ** forecast_steps) > abs(series[-1]) * 0.5:
            warnings.append({
                "type": AnalysisReason.TREND_EXPLOSION.value,
                "detail": f"Trend may cause unrealistic long-term forecasts"
            })
        
        return AnalysisResult.ok(
            {
                "forecast": forecast,
                "lower_bound": lower,
                "upper_bound": upper,
                "parameters": {
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                    "phi": phi if damp else 1.0,
                    "damped": damp
                }
            },
            warnings=warnings if warnings else None
        )
    
    except Exception as e:
        return AnalysisResult.error(
            AnalysisReason.UNEXPECTED_ERROR,
            str(e)
        )


def fit_exponential_decay(series: list) -> AnalysisResult:
    """
    拟合 y = a * exp(-t / tau) + c（修正版）。
    
    修正内容：
    - 根据序列趋势自适应选择初始参数
    - 添加收敛性诊断
    - 检查模型退化
    """
    series = np.array(series, dtype=float)
    t = np.arange(len(series), dtype=float)
    
    def model(t, a, tau, c):
        return a * np.exp(-t / tau) + c
    
    try:
        # 检测序列是收敛还是发散
        diffs = np.diff(series)
        recent_trend = np.mean(diffs[-len(diffs)//3:])
        early_trend = np.mean(diffs[:len(diffs)//3])
        is_converging = abs(recent_trend) < abs(early_trend) * 0.5
        
        if is_converging:
            # 收敛序列：a = 初始 - 收敛值
            p0 = [series[0] - series[-1], len(series) / 2, series[-1]]
            bounds = ([0, 1, -np.inf], [np.inf, len(series) * 2, np.inf])
        else:
            # 发散或波动序列：允许 a < 0
            p0 = [series[0] - series[-1], len(series) / 2, series[0]]
            bounds = ([-np.inf, 1, -np.inf], [np.inf, len(series) * 2, np.inf])
        
        popt, pcov = curve_fit(model, t, series, p0=p0, bounds=bounds, maxfev=5000)
        predicted = model(t, *popt)
        
        # 计算 R²
        ss_res = np.sum((series - predicted) ** 2)
        ss_tot = np.sum((series - np.mean(series)) ** 2)
        r_squared = 1 - ss_res / (ss_tot + 1e-10)
        
        # 诊断检查
        warnings = []
        
        # 检查模型退化
        if r_squared > 0.99 and popt[1] > len(series) * 10:
            warnings.append({
                "type": AnalysisReason.NEAR_PERFECT_FIT.value,
                "detail": f"Near-perfect fit but time constant ({popt[1]:.0f}) "
                         f"far exceeds series length ({len(series)}). Model likely degenerate."
            })
        
        # 检查收敛性
        convergence_value = popt[2]
        half_life = popt[1] * np.log(2)
        if half_life > len(series):
            warnings.append({
                "type": AnalysisReason.NON_CONVERGING_SERIES.value,
                "detail": f"Half-life ({half_life:.0f} steps) exceeds series length. "
                         f"Convergence not reliably established."
            })
        
        return AnalysisResult.ok(
            {
                "convergence_target": round(float(convergence_value), 4),
                "time_constant": round(float(popt[1]), 4),
                "half_life": round(float(half_life), 2),
                "r_squared": round(float(r_squared), 4),
                "is_converging": is_converging,
            },
            warnings=warnings if warnings else None
        )
    
    except Exception as e:
        return AnalysisResult.error(
            AnalysisReason.UNEXPECTED_ERROR,
            f"Exponential decay fit failed: {str(e)}"
        )


def compute_trend_forecast(buffer_data: dict, 
                           forecast_fields: list) -> AnalysisResult:
    """对核心指标做趋势预测"""
    results = {}
    status_by_field = {}
    all_warnings = []
    
    for field in forecast_fields:
        series = [snap.get(field) for snap in buffer_data.values()]
        series = [v for v in series if v is not None]
        
        if len(series) < 30:
            status_by_field[field] = {
                "status": AnalysisStatus.UNAVAILABLE.value,
                "reason": AnalysisReason.INSUFFICIENT_DATA.value
            }
            continue
        
        # Holt-Winters 预测
        hw_result = holt_winters_forecast(series)
        
        # 指数衰减拟合
        decay_result = fit_exponential_decay(series)
        
        results[field] = {
            "holt_winters": hw_result.to_dict() if hw_result.status == AnalysisStatus.OK else None,
            "exponential_decay": decay_result.to_dict() if decay_result.status == AnalysisStatus.OK else None,
        }
        
        # 收集状态
        field_status = AnalysisStatus.OK
        if hw_result.status != AnalysisStatus.OK and decay_result.status != AnalysisStatus.OK:
            field_status = AnalysisStatus.ERROR
        elif hw_result.status == AnalysisStatus.WARNING or decay_result.status == AnalysisStatus.WARNING:
            field_status = AnalysisStatus.WARNING
        
        status_by_field[field] = {
            "status": field_status.value,
            "holt_winters_status": hw_result.status.value,
            "exponential_decay_status": decay_result.status.value
        }
        
        # 收集警告
        if hw_result.warnings:
            all_warnings.extend([{**w, "field": field} for w in hw_result.warnings])
        if decay_result.warnings:
            all_warnings.extend([{**w, "field": field} for w in decay_result.warnings])
    
    overall_status = AnalysisStatus.OK
    if all(s["status"] == AnalysisStatus.ERROR.value for s in status_by_field.values()):
        overall_status = AnalysisStatus.ERROR
    elif any(s["status"] == AnalysisStatus.WARNING.value for s in status_by_field.values()):
        overall_status = AnalysisStatus.WARNING
    
    return AnalysisResult(
        overall_status,
        AnalysisReason.SUCCESS if overall_status == AnalysisStatus.OK else AnalysisReason.FEATURE_EXTRACTION_FAILED,
        data={
            "forecasts": results,
            "status_by_field": status_by_field
        },
        warnings=all_warnings if all_warnings else None
    )

# ============================================================
# 五、效果量化 (Mann-Whitney U)
# ============================================================

def compute_before_after(current_snapshot: dict, 
                         previous_snapshot: dict, 
                         all_fields: list) -> AnalysisResult:
    """
    对比两份快照中每个指标的最新 N 轮数据。
    使用 Mann-Whitney U 检验评估调参效果。
    """
    curr_buffer = current_snapshot.get("_buffer_data")
    prev_buffer = previous_snapshot.get("_buffer_data")
    
    if not curr_buffer or not prev_buffer:
        return AnalysisResult.unavailable(
            AnalysisReason.MISSING_BUFFER_DATA,
            "Missing _buffer_data in one or both snapshots"
        )
    
    try:
        curr_iters = sorted(curr_buffer.keys())
        prev_iters = sorted(prev_buffer.keys())
        
        curr_count = len(curr_iters)
        prev_count = len(prev_iters)
        window = min(curr_count, prev_count, 100)
        
        results = {}
        skipped = []
        
        for field in all_fields:
            curr_vals = [curr_buffer[it].get(field) for it in curr_iters[-window:]]
            prev_vals = [prev_buffer[it].get(field) for it in prev_iters[-window:]]
            curr_vals = [v for v in curr_vals if v is not None]
            prev_vals = [v for v in prev_vals if v is not None]
            
            if len(curr_vals) < 10 or len(prev_vals) < 10:
                skipped.append({
                    "field": field,
                    "reason": AnalysisReason.INSUFFICIENT_DATA.value,
                    "detail": f"curr: {len(curr_vals)}, prev: {len(prev_vals)}"
                })
                continue
            
            try:
                u_stat, p_val = mannwhitneyu(curr_vals, prev_vals, alternative="two-sided")
                results[field] = {
                    "before_mean": round(float(np.mean(prev_vals)), 4),
                    "after_mean": round(float(np.mean(curr_vals)), 4),
                    "u_statistic": round(float(u_stat), 2),
                    "p_value": round(float(p_val), 4),
                }
            except Exception as e:
                skipped.append({
                    "field": field,
                    "reason": AnalysisReason.UNEXPECTED_ERROR.value,
                    "detail": str(e)
                })
        
        warnings = []
        if skipped:
            warnings.append({
                "type": "skipped_fields",
                "count": len(skipped),
                "details": skipped[:5]
            })
        
        if not results:
            return AnalysisResult.error(
                AnalysisReason.FEATURE_EXTRACTION_FAILED,
                "No fields could be compared"
            )
        
        return AnalysisResult.ok(results, warnings=warnings if warnings else None)
    
    except Exception as e:
        return AnalysisResult.error(
            AnalysisReason.UNEXPECTED_ERROR,
            str(e)
        )

# ============================================================
# 主入口
# ============================================================

def execute(snapshot_path: str, data_dir: Optional[str] = None, _control: dict = None) -> dict:
    """
    读取快照完整 JSON，产出五类深度分析结果。
    
    Returns:
        包含所有分析结果的字典，每个分析结果都带有 status/status/reason/data 结构
    """
    try:
        current = _load_snapshot(snapshot_path)
    except Exception as e:
        return {
            "error": {
                "status": AnalysisStatus.ERROR.value,
                "reason": AnalysisReason.UNEXPECTED_ERROR.value,
                "detail": f"读取快照失败: {e}"
            }
        }
    
    buffer_data = current.get("_buffer_data")
    if not buffer_data:
        return {
            "error": {
                "status": AnalysisStatus.ERROR.value,
                "reason": AnalysisReason.MISSING_BUFFER_DATA.value,
                "detail": "快照中缺少 _buffer_data。请确保 parse_log 在完整 JSON 中保存了 buffer 原始数据"
            }
        }
    
    # 提取所有字段
    all_fields = list(buffer_data[next(iter(buffer_data.keys()))].keys())
    
    # 默认的趋势预测字段
    forecast_fields = [
        "Metrics/base_velocity/error_vel_xy",
        "Curriculum/terrain_levels",
        "Episode_Reward/feet_gait",
        "Episode_Reward/track_lin_vel_xy_exp",
    ]
    # 只保留实际存在的字段
    forecast_fields = [f for f in forecast_fields if f in all_fields]
    
    # 执行各项分析
    result = {
        "change_points": compute_all_change_points(buffer_data, all_fields).to_dict(),
        "event_links": compute_all_event_links(buffer_data, all_fields).to_dict(),
        "anomalies": compute_anomalies(buffer_data, all_fields).to_dict(),
        "trend_forecast": compute_trend_forecast(buffer_data, forecast_fields).to_dict(),
    }
    
    # 效果量化（需要历史快照）
    if data_dir:
        previous = _find_previous_snapshot(data_dir, snapshot_path)
        if previous:
            result["before_after"] = compute_before_after(
                current, previous, all_fields
            ).to_dict()
        else:
            result["before_after"] = AnalysisResult.unavailable(
                AnalysisReason.PREVIOUS_SNAPSHOT_NOT_FOUND,
                "No previous snapshot found in data_dir"
            ).to_dict()
    else:
        result["before_after"] = AnalysisResult.unavailable(
            AnalysisReason.INSUFFICIENT_DATA,
            "data_dir not provided, cannot compute before/after comparison"
        ).to_dict()
    
    if _control is not None:
        _control["_pending_rebuttal"] = True
    return _clean_for_json(result)

# ============================================================
# CLI Entry Point
# ============================================================

MAX_OUTPUT_CHARS = 200_000

def main():
    parser = argparse.ArgumentParser(
        description="Deep dig training data (v2.0 - Statistical Rigor Edition)"
    )
    parser.add_argument("--snapshot_path", required=True, 
                       help="当前快照的完整 JSON 文件路径")
    parser.add_argument("--data_dir", default=None,
                       help="快照目录。需要效果量化时传入")
    args = parser.parse_args()
    
    try:
        result = execute(args.snapshot_path, args.data_dir)
        output_str = json.dumps(result, indent=2, ensure_ascii=False)
        if len(output_str) > MAX_OUTPUT_CHARS:
            output_str = output_str[:MAX_OUTPUT_CHARS] + "\n... [truncated]"
        print(output_str)
    except Exception as e:
        error_output = {
            "error": {
                "status": AnalysisStatus.ERROR.value,
                "reason": AnalysisReason.UNEXPECTED_ERROR.value,
                "detail": str(e)
            }
        }
        print(json.dumps(error_output, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()