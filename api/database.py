import sqlite3
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "bugs.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # sonuçları dict gibi okuyabilmek için
    return conn


def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bugs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT,
            schema_type TEXT,
            component TEXT,
            trigger_action TEXT,
            error_signature TEXT,
            severity TEXT,
            severity_confidence REAL,
            needs_human_review BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def insert_bug(description: str, result: dict) -> int:
    conn = get_connection()
    cursor = conn.execute(
        """
        INSERT INTO bugs (
            description, schema_type, component, trigger_action,
            error_signature, severity, severity_confidence, needs_human_review
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            description,
            result.get("schema_type"),
            result.get("component"),
            result.get("trigger_action"),
            result.get("error_signature"),
            result.get("severity"),
            result.get("severity_confidence"),
            result.get("needs_human_review"),
        ),
    )
    conn.commit()
    bug_id = cursor.lastrowid
    conn.close()
    return bug_id


def get_bugs(severity: str = None, needs_review: bool = None, schema_type: str = None, limit: int = 100):
    conn = get_connection()
    query = "SELECT * FROM bugs WHERE 1=1"
    params = []

    if severity is not None:
        query += " AND severity = ?"
        params.append(severity)

    if needs_review is not None:
        query += " AND needs_human_review = ?"
        params.append(int(needs_review))

    if schema_type is not None:
        query += " AND schema_type = ?"
        params.append(schema_type)

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_stats():
    conn = get_connection()

    total = conn.execute("SELECT COUNT(*) as count FROM bugs").fetchone()["count"]

    severity_dist = conn.execute("""
        SELECT severity, COUNT(*) as count
        FROM bugs GROUP BY severity ORDER BY count DESC
    """).fetchall()

    schema_dist = conn.execute("""
        SELECT schema_type, COUNT(*) as count
        FROM bugs GROUP BY schema_type ORDER BY count DESC
    """).fetchall()

    review_count = conn.execute("""
        SELECT COUNT(*) as count FROM bugs WHERE needs_human_review = 1
    """).fetchone()["count"]

    avg_confidence = conn.execute("""
        SELECT AVG(severity_confidence) as avg_conf FROM bugs
    """).fetchone()["avg_conf"]

    conn.close()

    return {
        "total_bugs": total,
        "severity_distribution": {row["severity"]: row["count"] for row in severity_dist},
        "schema_type_distribution": {row["schema_type"]: row["count"] for row in schema_dist},
        "needs_review_count": review_count,
        "needs_review_percentage": round(review_count / total * 100, 1) if total > 0 else 0,
        "average_severity_confidence": round(avg_confidence, 4) if avg_confidence else None,
    }

def delete_bug(bug_id: int) -> bool:
    conn = get_connection()
    cursor = conn.execute("DELETE FROM bugs WHERE id = ?", (bug_id,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted