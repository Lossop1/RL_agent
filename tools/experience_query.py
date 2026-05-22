import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "state", "experience.db")

TOOL = {
    "name": "experience_query",
    "description": "查询历史训练知识库。支持按 run_id 查同次训练链条，或按任务特征查相似历史案例。",
    "parameters": {
        "run_id": {"type": "str", "required": False, "desc": "按训练闭环查询。传入时忽略其他过滤条件"},
        "robot": {"type": "str", "required": False, "desc": "机器人名"},
        "terrain": {"type": "str", "required": False, "desc": "地形类型"},
        "dominant_penalty": {"type": "str", "required": False},
        "focus_quality": {"type": "str", "required": False},
        "error_vel_xy": {"type": "float", "required": False},
        "entropy": {"type": "float", "required": False},
        "limit": {"type": "int", "required": False, "desc": "返回数量上限，默认10"}
    }
}

# 聚类映射
PENALTY_CLUSTERS = {
    "joint_acc_l2": "smoothness",
    "action_rate_l2": "smoothness",
    "joint_torques_l2": "energy_efficiency",
    "joint_power": "energy_efficiency",
    "feet_slide": "gait_quality",
    "feet_gait": "gait_quality",
    "feet_air_time": "gait_quality",
    "feet_air_time_variance": "gait_quality",
    "joint_pos_penalty": "constraint_violation",
    "joint_pos_limits": "constraint_violation",
    "undesired_contacts": "constraint_violation",
    "contact_forces": "constraint_violation",
    "ang_vel_xy_l2": "stability",
    "lin_vel_z_l2": "stability",
    "upward": "stability",
    "stand_still": "velocity_tracking",
    "joint_mirror": "gait_quality",
}

def _init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS training_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            robot TEXT NOT NULL,
            terrain TEXT NOT NULL,
            run_id TEXT NOT NULL,
            round INTEGER NOT NULL,
            iteration INTEGER NOT NULL,
            reward_config TEXT,
            diagnosis TEXT NOT NULL,
            dominant_penalty TEXT,
            dominant_penalty_share REAL,
            focus_quality TEXT,
            error_vel_xy REAL,
            entropy REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(run_id, round)
        );
    """)
    conn.commit()
    conn.close()

def execute(run_id: str = None, robot: str = None, terrain: str = None,
            dominant_penalty: str = None, focus_quality: str = None,
            error_vel_xy: float = None, entropy: float = None,
            limit: int = 10) -> dict:
    _init_db()
    
    if not os.path.exists(DB_PATH):
        return {"status": "ok", "cases": [], "total": 0}
    
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        
        if run_id:
            # 模式一：查同次训练链条
            rows = conn.execute(
                """SELECT * FROM training_records 
                   WHERE run_id = ? 
                   ORDER BY round""",
                (run_id,)
            ).fetchall()
        else:
            # 模式二：相似度查询
            if not robot or not terrain:
                return {"status": "error", "message": "相似度查询需要 robot 和 terrain"}
            
            # 计算相似度并排序
            current_cluster = PENALTY_CLUSTERS.get(dominant_penalty, "") if dominant_penalty else ""
            
            rows = conn.execute(
                f"""SELECT *,
                    (CASE WHEN dominant_penalty = ? THEN 3
                          WHEN dominant_penalty IN (
                              SELECT key FROM (
                                  SELECT 'joint_acc_l2' AS key UNION SELECT 'action_rate_l2'
                              ) WHERE key IN (
                                  SELECT CASE WHEN ? IN ('joint_acc_l2','action_rate_l2') THEN 
                                      CASE WHEN dominant_penalty IN ('joint_acc_l2','action_rate_l2') THEN dominant_penalty END
                                  END
                              )
                          ) THEN 1
                          ELSE 0
                     END
                     + CASE WHEN focus_quality = ? THEN 2 ELSE 0 END
                     + CASE WHEN ABS(error_vel_xy - ?) < 0.3 THEN 1 ELSE 0 END
                     + CASE WHEN ? IS NOT NULL AND entropy IS NOT NULL AND ABS(entropy - ?) < 5 THEN 1 ELSE 0 END
                    ) AS score
                    FROM training_records
                    WHERE robot = ? AND terrain = ?
                    ORDER BY score DESC
                    LIMIT ?""",
                (dominant_penalty, dominant_penalty or "", focus_quality,
                 error_vel_xy or 0, entropy, entropy,
                 robot, terrain, limit)
            ).fetchall()
        
        conn.close()
        
        cases = []
        for row in rows:
            case = dict(row)
            for field in ["reward_config", "diagnosis"]:
                if case.get(field) and isinstance(case[field], str):
                    try:
                        case[field] = json.loads(case[field])
                    except json.JSONDecodeError:
                        pass
            cases.append(case)
        
        return {"status": "ok", "cases": cases, "total": len(cases)}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}