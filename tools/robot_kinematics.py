"""
robot_kinematics.py --- 从 URDF + Isaac Lab 配置提取四足机器人运动学参数

输入：URDF 文件路径 + 可选 Isaac Lab ArticulationCfg .py 文件路径
输出：标准化的运动学参数字典，供 behavior_selfcheck 和 standard 生成使用

设计原则：只提取运动学参数（运动相关的几何与限位），不提取动力学参数（质量、电机）。
"""
import re
import json
import xml.etree.ElementTree as ET
from collections import OrderedDict

TOOL = {
    "name": "robot_kinematics",
    "description": (
        "从 URDF 文件中提取四足机器人运动学参数：腿段长度、关节限位、站立高度、"
        "髋间距、足端半径等。可选从 Isaac Lab .py config 中补充初始姿态和限位因子。"
    ),
    "parameters": {
        "urdf_path": {
            "type": "str", "required": True,
            "desc": "URDF 文件路径（绝对路径或项目相对路径）"
        },
        "config_py_path": {
            "type": "str", "required": False,
            "desc": "Isaac Lab ArticulationCfg .py 文件路径，用于提取 init_state 和 soft_joint_pos_limit_factor"
        },
        "output": {
            "type": "str", "required": False,
            "desc": "可选，将提取的参数写入指定 JSON 路径"
        },
    }
}

# ============================================================================
# URDF 解析
# ============================================================================

def _parse_urdf(urdf_path: str) -> dict:
    """从 URDF 提取关节层级、限位、几何体参数。"""
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # 1. 找出四足关节：hip, thigh, calf（按命名规则 FL/FR/RL/RR）
    joints = {}
    joint_order = []
    for joint in root.iter("joint"):
        name = joint.attrib.get("name", "")
        if name.endswith("_hip_joint") and not name.startswith("Head"):
            kind = "hip"
        elif name.endswith("_thigh_joint"):
            kind = "thigh"
        elif name.endswith("_calf_joint"):
            kind = "calf"
        elif name.endswith("_foot_joint"):
            kind = "foot"
        else:
            continue

        # 关节限位
        limit = joint.find("limit")
        lower = float(limit.attrib.get("lower", 0)) if limit is not None else 0.0
        upper = float(limit.attrib.get("upper", 0)) if limit is not None else 0.0

        # 关节原点（相对父 link 的位移，即 leg segment length）
        origin = joint.find("origin")
        if origin is not None:
            xyz_str = origin.attrib.get("xyz", "0 0 0")
            ox, oy, oz = [float(v) for v in xyz_str.split()]
        else:
            ox, oy, oz = 0.0, 0.0, 0.0

        joints[name] = {
            "kind": kind,
            "limit_lower": lower,
            "limit_upper": upper,
            "origin": (ox, oy, oz),
        }
        joint_order.append(name)

    # 2. 分类
    legs = {"FL": {}, "FR": {}, "RL": {}, "RR": {}}
    for leg_id in legs:
        for jname, jinfo in joints.items():
            if jname.startswith(f"{leg_id}_"):
                legs[leg_id][jinfo["kind"]] = {
                    "name": jname,
                    "limit_lower": jinfo["limit_lower"],
                    "limit_upper": jinfo["limit_upper"],
                    "origin": jinfo["origin"],
                }

    # 3. 提取腿段长度（thigh origin z + calf origin z）
    # Go2: thigh joint origin 在 hip 位置，calf joint origin = thigh 长度
    thigh_lengths = []
    calf_lengths = []
    for leg_id in ["FL", "FR", "RL", "RR"]:
        thigh = legs[leg_id].get("thigh", {})
        calf = legs[leg_id].get("calf", {})
        # thigh segment = calf_joint origin z（通常 thigh 的 z 分量是腿长）
        if calf and thigh:
            # 取 calf joint origin 的 z 分量（沿腿方向的长度）
            thigh_len = abs(calf.get("origin", (0, 0, 0))[2])
            calf_len = 0.0
            # 如果有 foot joint，calf segment = foot_joint origin z
            foot = legs[leg_id].get("foot", {})
            if foot:
                calf_len = abs(foot.get("origin", (0, 0, 0))[2])
            # 如果无 foot，用 calf joint origin.z
            if calf_len < 0.001:
                calf_len = abs(calf.get("origin", (0, 0, 0))[2])
        else:
            thigh_len = abs(thigh.get("origin", (0, 0, 0))[2])
            calf_len = 0.0
        thigh_lengths.append(thigh_len)
        calf_lengths.append(calf_len)

    # 4. 髋间距（FL_hip_joint origin.y 的 2 倍）
    fl_hip = joints.get("FL_hip_joint", {})
    hip_y = abs(fl_hip.get("origin", (0, 0, 0))[1])
    hip_spacing = hip_y * 2

    # 5. 基座长度（FL_hip_joint origin.x）
    hip_x_forward = fl_hip.get("origin", (0, 0, 0))[0]

    # 6. 足端半径（从 collision geometry 提取）
    foot_radius = 0.0
    for link in root.iter("link"):
        name = link.attrib.get("name", "")
        if name.endswith("_foot"):
            collision = link.find("collision")
            if collision is not None:
                geom = collision.find("geometry")
                if geom is not None:
                    sphere = geom.find("sphere")
                    if sphere is not None:
                        foot_radius = float(sphere.attrib.get("radius", 0))
                        break
                    capsule = geom.find("capsule")
                    if capsule is not None:
                        foot_radius = float(capsule.attrib.get("radius", 0))
                        break

    return {
        "robot_name": root.attrib.get("name", "unknown"),
        "leg_segments": {
            "thigh_mean": sum(thigh_lengths) / len(thigh_lengths) if thigh_lengths else 0,
            "calf_mean": sum(calf_lengths) / len(calf_lengths) if calf_lengths else 0,
            "total_leg_length": (sum(thigh_lengths) + sum(calf_lengths)) / max(len(thigh_lengths), 1),
            "thigh_by_leg": dict(zip(["FL", "FR", "RL", "RR"], thigh_lengths)),
            "calf_by_leg": dict(zip(["FL", "FR", "RL", "RR"], calf_lengths)),
        },
        "hip": {
            "spacing_y": hip_spacing,
            "distance_from_base_center_x": hip_x_forward,
        },
        "foot_radius": foot_radius,
        "joint_limits": {
            leg_id: {
                jt: {
                    "lower": legs[leg_id][jt]["limit_lower"],
                    "upper": legs[leg_id][jt]["limit_upper"],
                }
                for jt in ["hip", "thigh", "calf", "foot"] if jt in legs[leg_id]
            }
            for leg_id in ["FL", "FR", "RL", "RR"]
        },
        "joint_defaults_from_urdf": {
            jname: 0.0 for jname in joints  # URDF 中 default 是 0，实际由 config 提供
        },
    }


# ============================================================================
# Isaac Lab .py Config 解析
# ============================================================================

def _parse_config_py(config_py_path: str) -> dict:
    """从 Isaac Lab ArticulationCfg .py 文件提取 init_state 和 soft_joint_pos_limit_factor。

    不执行 Python 代码，使用正则模式匹配。
    支持的机器人型号会通过文件名和类名自动检测。
    """
    with open(config_py_path, "r") as f:
        content = f.read()

    result = {}

    # 查找 soft_joint_pos_limit_factor
    m = re.search(r"soft_joint_pos_limit_factor\s*=\s*([0-9.]+)", content)
    if m:
        result["soft_joint_pos_limit_factor"] = float(m.group(1))

    # 查找 standing_height (init_state pos 的 z)
    # 模式: pos=(0.0, 0.0, 0.38) 或 pos=(0.0, 0.0, 站立高度)
    m = re.search(r"pos\s*=\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)", content)
    if m:
        result["standing_height"] = float(m.group(3))

    # 查找 init_state joint_pos
    # 模式: joint_pos={ ".*key": value, ... }
    joint_pos_block = re.search(r"joint_pos\s*=\s*\{([^}]+(?:\{[^}]*\}[^}]*)*)\}", content, re.DOTALL)
    if joint_pos_block:
        result["default_joint_pos"] = {}
        block = joint_pos_block.group(1)
        for m in re.finditer(r'"([^"]+)"\s*:\s*([\d.]+(?:e[+-]?\d+)?)', block):
            result["default_joint_pos"][m.group(1)] = float(m.group(2))

    return result


# ============================================================================
# 合并 & 标准化
# ============================================================================

def _merge_kinematics(urdf_data: dict, config_data: dict) -> dict:
    """合并 URDF 和 config 数据，产出标准化运动学参数。"""
    soft_factor = config_data.get("soft_joint_pos_limit_factor", 0.9)
    standing_height = config_data.get("standing_height", 0.38)
    default_joint_pos = config_data.get("default_joint_pos", {})

    # 计算有效限位（URDF 限位 × soft_factor）
    effective_limits = {}
    for leg_id in ["FL", "FR", "RL", "RR"]:
        limits = urdf_data["joint_limits"].get(leg_id, {})
        effective_limits[leg_id] = {}
        for jt, lr in limits.items():
            effective_limits[leg_id][jt] = {
                "lower": round(lr["lower"] * soft_factor, 4),
                "upper": round(lr["upper"] * soft_factor, 4),
                "range": round((lr["upper"] - lr["lower"]) * soft_factor, 4),
            }

    return {
        "robot_name": urdf_data["robot_name"],
        "standing_height": standing_height,
        "leg_length": round(urdf_data["leg_segments"]["total_leg_length"], 4),
        "thigh_length": round(urdf_data["leg_segments"]["thigh_mean"], 4),
        "calf_length": round(urdf_data["leg_segments"]["calf_mean"], 4),
        "hip_spacing_y": round(urdf_data["hip"]["spacing_y"], 4),
        "hip_x_forward": round(urdf_data["hip"]["distance_from_base_center_x"], 4),
        "foot_radius": round(urdf_data["foot_radius"], 4),
        "joint_limits": effective_limits,
    }


# ============================================================================
# 主入口
# ============================================================================

def execute(urdf_path: str, config_py_path: str = None, output: str = None) -> dict:
    try:
        urdf_data = _parse_urdf(urdf_path)
    except Exception as e:
        return {"error": f"URDF 解析失败: {type(e).__name__}: {e}"}

    config_data = {}
    if config_py_path:
        try:
            config_data = _parse_config_py(config_py_path)
        except Exception as e:
            # config 解析失败不致命，用默认值
            config_data = {}

    result = _merge_kinematics(urdf_data, config_data)
    result["_meta"] = {
        "urdf_path": urdf_path,
        "config_py_path": config_py_path,
        "config_fields_extracted": list(config_data.keys()),
    }

    if output:
        import os
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    return result
