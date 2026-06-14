# =============================================================
#  employee_db.py  —  SQLAlchemy persistence layer for employee
#                     face embeddings (InsightFace ArcFace 512-d)
#
#  Schema (table: employees)
#  ─────────────────────────────────────────────────────────────
#  employee_id          TEXT PRIMARY KEY   e.g. "kalyan_sai"
#  employee_name        TEXT               display name
#  embedding            BLOB               512-dim float32 array
#  image_count          INTEGER            usable images averaged
#  enrollment_timestamp TEXT               ISO-8601 UTC
# =============================================================

import logging
import os
from datetime import datetime, timezone
import numpy as np

from sqlalchemy import create_engine, text, MetaData, Table, Column, String, Integer, LargeBinary
from config import EMPLOYEE_DB_URL, EMPLOYEE_DB_PATH

log = logging.getLogger(__name__)

EMBEDDING_DIM = 512

class EmployeeDB:
    def __init__(self, db_url: str = EMPLOYEE_DB_URL):
        self.db_url = db_url
        if self.db_url.startswith("sqlite:///"):
            os.makedirs(os.path.dirname(EMPLOYEE_DB_PATH), exist_ok=True)
            
        self.engine = create_engine(self.db_url)
        self.metadata = MetaData()
        
        self.employees = Table(
            'employees', self.metadata,
            Column('employee_id', String, primary_key=True),
            Column('employee_name', String, nullable=False),
            Column('embedding', LargeBinary, nullable=False),
            Column('image_count', Integer, nullable=False, default=0),
            Column('enrollment_timestamp', String, nullable=False)
        )

    @staticmethod
    def _emb_to_blob(embedding: np.ndarray) -> bytes:
        """Serialize a float32 numpy array to raw bytes."""
        return embedding.astype(np.float32).tobytes()

    @staticmethod
    def _blob_to_emb(blob: bytes) -> np.ndarray:
        """Deserialize raw bytes back to a float32 numpy array."""
        arr = np.frombuffer(blob, dtype=np.float32).copy()
        return arr.reshape(-1, 512)

    def initialize(self):
        """Create the `employees` table if it does not exist."""
        self.metadata.create_all(self.engine)
        log.debug("[EmployeeDB] Table initialized at %s", self.db_url)

    def upsert(
        self,
        employee_id: str,
        employee_name: str,
        embedding: np.ndarray,
        image_count: int,
    ):
        if len(embedding.shape) != 2 or embedding.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding must be shape (N, {EMBEDDING_DIM}), got {embedding.shape}"
            )

        ts = datetime.now(timezone.utc).isoformat()
        blob = self._emb_to_blob(embedding)
        
        with self.engine.begin() as conn:
            chk_query = text("SELECT employee_id FROM employees WHERE employee_id = :id")
            row = conn.execute(chk_query, {"id": employee_id}).fetchone()
            
            if row:
                upd = text("""
                    UPDATE employees 
                    SET employee_name = :name, embedding = :emb, image_count = :cnt, enrollment_timestamp = :ts
                    WHERE employee_id = :id
                """)
                conn.execute(upd, {"name": employee_name, "emb": blob, "cnt": image_count, "ts": ts, "id": employee_id})
            else:
                ins = self.employees.insert().values(
                    employee_id=employee_id, employee_name=employee_name,
                    embedding=blob, image_count=image_count, enrollment_timestamp=ts
                )
                conn.execute(ins)
                
        log.info("[EmployeeDB] Upserted %s (%s) — %d images", employee_id, employee_name, image_count)

    def delete(self, employee_id: str) -> bool:
        with self.engine.begin() as conn:
            cur = conn.execute(text("DELETE FROM employees WHERE employee_id = :id"), {"id": employee_id})
            deleted = cur.rowcount > 0
            
        if deleted:
            log.info("[EmployeeDB] Deleted employee: %s", employee_id)
        else:
            log.warning("[EmployeeDB] Delete: employee not found: %s", employee_id)
        return deleted

    def get_all(self) -> dict:
        result = {}
        with self.engine.connect() as conn:
            rows = conn.execute(text("SELECT employee_id, employee_name, embedding, image_count FROM employees")).fetchall()
            for row in rows:
                result[row.employee_id] = {
                    "name":        row.employee_name,
                    "embeddings":  list(self._blob_to_emb(row.embedding)),
                    "image_count": row.image_count,
                }
        log.info("[EmployeeDB] Loaded %d employee(s) from %s", len(result), self.db_url)
        return result

    def get_one(self, employee_id: str) -> dict | None:
        with self.engine.connect() as conn:
            row = conn.execute(text("SELECT * FROM employees WHERE employee_id = :id"), {"id": employee_id}).fetchone()
            
        if row is None:
            return None
            
        return {
            "employee_id":          row.employee_id,
            "employee_name":        row.employee_name,
            "embedding":            list(self._blob_to_emb(row.embedding)),
            "image_count":          row.image_count,
            "enrollment_timestamp": row.enrollment_timestamp,
        }

    def list_employees(self) -> list:
        with self.engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT employee_id, employee_name, image_count, enrollment_timestamp
                FROM employees
                ORDER BY employee_name
            """)).fetchall()
        return [dict(row._mapping) for row in rows]

    def count(self) -> int:
        with self.engine.connect() as conn:
            return conn.execute(text("SELECT COUNT(*) FROM employees")).fetchone()[0]

    def employee_exists(self, employee_id: str) -> bool:
        with self.engine.connect() as conn:
            row = conn.execute(text("SELECT 1 FROM employees WHERE employee_id = :id"), {"id": employee_id}).fetchone()
        return row is not None
