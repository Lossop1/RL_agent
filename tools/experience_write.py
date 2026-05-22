import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "state", "experience.db")

TOOL = {
    "name": "experience_write",
    "description": "向训练知识库写入诊断记录。每次诊断后必须调用。",
    "parameters": {
        "robot": {"type": "str", "required": True},
        "terrain": {"type": "str", "required": True},
        "run_id": {"type": "str", "required": True},
        "round": {"type": "int", "required": True},
        "iteration": {"type": "int", "required": True},
        "reward_config": {"type": "object", "required": True, "desc": "本轮修改的奖励配置项，值为新的绝对值。未修改时传 null"},
        "diagnosis": {"type": "object", "required": True, "desc": "诊断结论 JSON，包含 behavior_profile, root_cause, action_decided, action_detail"},
        "dominant_penalty": {"type": "str", "required": False},
        "dominant_penalty_share": {"type": "float", "required": False},
        "focus_quality": {"type": "str", "required": False},
        "error_vel_xy": {"type": "float", "required": False},
        "entropy": {"type": "float", "required": False}
    }
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

def execute(robot: str, terrain: str, run_id: str, round: int, iteration: int,
            reward_config: dict, diagnosis: dict,
            dominant_penalty: str = None, dominant_penalty_share: float = None,
            focus_quality: str = None, error_vel_xy: float = None,
            entropy: float = None) -> dict:
    _init_db()
    
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # 序列化 JSON 字段
        reward_config_str = json.dumps(reward_config, ensure_ascii=False) if reward_config else None
        diagnosis_str = json.dumps(diagnosis, ensure_ascii=False)
        
        # 尝试 UPDATE，如果不存在则 INSERT
        cursor = conn.execute(
            """UPDATE training_records 
               SET iteration = ?, reward_config = ?, diagnosis = ?,
                   dominant_penalty = ?, dominant_penalty_share = ?,
                   focus_quality = ?, error_vel_xy = ?, entropy = ?
               WHERE run_id = ? AND round = ?""",
            (iteration, reward_config_str, diagnosis_str,
             dominant_penalty, dominant_penalty_share,
             focus_quality, error_vel_xy, entropy,
             run_id, round)
        )
        
        if cursor.rowcount == 0:
            conn.execute(
                """INSERT INTO training_records 
                   (robot, terrain, run_id, round, iteration, reward_config, diagnosis,
                    dominant_penalty, dominant_penalty_share, focus_quality, error_vel_xy, entropy)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (robot, terrain, run_id, round, iteration, reward_config_str, diagnosis_str,
                 dominant_penalty, dominant_penalty_share, focus_quality, error_vel_xy, entropy)
            )
        
        conn.commit()
        conn.close()
        return {"status": "ok"}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}