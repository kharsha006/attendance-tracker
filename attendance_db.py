# =============================================================
#  attendance_db.py  —  All database operations
#  Creates and manages the attendance database via SQLAlchemy.
# =============================================================

import os
from datetime import datetime, date
import pandas as pd
import pickle
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Float, UniqueConstraint
from tz_utils import strftime_today, strftime_now
from config import (
    DATABASE_PATH, MONTHLY_WORKING_DAYS, EMBEDDINGS_FILE,
    WORKDAY_START_TIME, WORKDAY_END_TIME, TARGET_WORK_HOURS,
    ATTENDANCE_DB_URL
)

# Initialize SQLAlchemy Engine
engine = None

def get_engine():
    global engine
    if engine is None:
        if ATTENDANCE_DB_URL.startswith("sqlite:///"):
            os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
        engine = create_engine(ATTENDANCE_DB_URL)
    return engine

metadata = MetaData()

attendance_events = Table(
    'attendance_events', metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('employee_id', String, nullable=False),
    Column('employee_name', String, nullable=False),
    Column('event_type', String, nullable=False),
    Column('event_time', String, nullable=False),
    Column('work_date', String, nullable=False),
    Column('camera_source', String, nullable=False),
    Column('confidence', Float, nullable=False)
)

daily_summary = Table(
    'daily_summary', metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('employee_id', String, nullable=False),
    Column('employee_name', String, nullable=False),
    Column('work_date', String, nullable=False),
    Column('first_seen', String),
    Column('last_seen', String),
    Column('hours_worked', Float),
    Column('status', String),
    Column('session_breakdown', String),
    UniqueConstraint('employee_id', 'work_date', name='uq_emp_date')
)

camera_status = Table(
    'camera_status', metadata,
    Column('camera_name', String, primary_key=True),
    Column('status', String, nullable=False),
    Column('last_seen', String),
    Column('downtime_start', String)
)

unknown_detections = Table(
    'unknown_detections', metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('timestamp', String, nullable=False),
    Column('camera_source', String, nullable=False),
    Column('image_name', String, nullable=False)
)


def initialise_database():
    """Create the attendance tables if they don't exist yet."""
    metadata.create_all(get_engine())
    print("[DB] Database initialised successfully via SQLAlchemy.")


def log_event(employee_id, employee_name, event_type, confidence, camera_source):
    event_time = strftime_now()
    work_date  = strftime_today()

    with get_engine().begin() as conn:
        conn.execute(
            attendance_events.insert().values(
                employee_id=employee_id,
                employee_name=employee_name,
                event_type=event_type,
                event_time=event_time,
                work_date=work_date,
                camera_source=camera_source,
                confidence=confidence
            )
        )

    print(f"[DB] Logged: {employee_name} | {event_type.upper()} | {event_time} | confidence: {confidence:.2f}")
    update_daily_summary(employee_id, employee_name, work_date)


def log_presence_event(employee_id, employee_name, confidence, camera_source):
    event_time = strftime_now()
    work_date  = strftime_today()

    with get_engine().begin() as conn:
        chk_query = text("SELECT id FROM daily_summary WHERE employee_id = :emp AND work_date = :wd")
        chk_row = conn.execute(chk_query, {"emp": employee_id, "wd": work_date}).fetchone()

        if chk_row:
            # Already marked today. Update last_seen.
            upd = text("""
                UPDATE daily_summary SET 
                    last_seen=:ls
                WHERE id=:id
            """)
            conn.execute(upd, {"ls": event_time, "id": chk_row.id})
            print(f"[DB] Presence Updated (Last Seen): {employee_name} | {event_time}")
        else:
            # First detection of the day
            ins = daily_summary.insert().values(
                employee_id=employee_id, employee_name=employee_name, work_date=work_date,
                first_seen=event_time, last_seen=event_time, status="Present",
                hours_worked=0.0, session_breakdown=None
            )
            conn.execute(ins)
            print(f"[DB] Attendance Marked (First Seen): {employee_name} | {event_time}")


def get_last_event_type_today(employee_id):
    today = strftime_today()
    with get_engine().connect() as conn:
        query = text("""
            SELECT event_type FROM attendance_events
            WHERE employee_id = :emp AND work_date = :td
            ORDER BY event_time DESC LIMIT 1
        """)
        row = conn.execute(query, {"emp": employee_id, "td": today}).fetchone()
        return row.event_type if row else None





def get_today_summary():
    today = strftime_today()
    with get_engine().connect() as conn:
        query = text("""
            SELECT employee_name, first_seen, last_seen, status
            FROM daily_summary
            WHERE work_date = :td
        """)
        rows = conn.execute(query, {"td": today}).fetchall()

    def sort_key(row):
        return row.first_seen if row.first_seen else "9999-99-99 99:99:99"
    rows = sorted(rows, key=sort_key)

    result = []
    for row in rows:
        first_time = row.first_seen
        last_time  = row.last_seen
        if first_time:
            first_time = datetime.strptime(first_time, "%Y-%m-%d %H:%M:%S").strftime("%I:%M %p")
        if last_time:
            last_time = datetime.strptime(last_time, "%Y-%m-%d %H:%M:%S").strftime("%I:%M %p")

        result.append({
            "name":         row.employee_name,
            "entry":        first_time or "—",
            "exit":         last_time or "—",
            "status":       row.status or "Absent"
        })

    return result

def get_date_summary(target_date: str):
    with get_engine().connect() as conn:
        query = text("""
            SELECT employee_name, first_seen, last_seen, status
            FROM daily_summary
            WHERE work_date = :td
        """)
        rows = conn.execute(query, {"td": target_date}).fetchall()
        
        def sort_key(row):
            return row.first_seen if row.first_seen else "9999-99-99 99:99:99"
        rows = sorted(rows, key=sort_key)
        
        return [dict(row._mapping) for row in rows]


def export_to_excel(start_date: str, end_date: str, filepath: str):
    with get_engine().connect() as conn:
        query = text("""
            SELECT employee_name AS "Employee", work_date AS "Date",
                   first_seen AS "First Seen", last_seen AS "Last Seen", status AS "Status"
            FROM daily_summary
            WHERE work_date BETWEEN :sd AND :ed
            ORDER BY work_date, employee_name
        """)
        df = pd.read_sql_query(query, conn, params={"sd": start_date, "ed": end_date})

    for col in ["First Seen", "Last Seen"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").strftime("%I:%M %p") if pd.notna(x) and x != "" else "—")

    df.to_excel(filepath, index=False)
    print(f"[DB] Exported to {filepath}")
    return filepath


def get_registered_employees():
    from config import EMPLOYEE_DB_URL
    import sqlalchemy
    
    try:
        emp_engine = sqlalchemy.create_engine(EMPLOYEE_DB_URL)
        with emp_engine.connect() as conn:
            rows = conn.execute(text("SELECT employee_id, employee_name FROM employees")).fetchall()
            if rows:
                return {r.employee_id: r.employee_name for r in rows}
    except Exception as e:
        print(f"[DB] Could not load from EMPLOYEE_DB_URL: {e}")
        pass

    try:
        with open(EMBEDDINGS_FILE, "rb") as f:
            data = pickle.load(f)
        return {emp_id: emp_data["name"] for emp_id, emp_data in data.items()}
    except Exception:
        return {}


def get_monthly_report(month_str: str):
    employees = get_registered_employees()
    report = []
    
    with get_engine().connect() as conn:
        for emp_id, emp_name in employees.items():
            query = text("""
                SELECT COUNT(*) FROM daily_summary
                WHERE employee_id = :emp AND work_date LIKE :md AND status IN ('Complete', 'Short', 'In office', 'Late Entry', 'Early Exit', 'Late & Early', 'Present')
            """)
            present_days = conn.execute(query, {"emp": emp_id, "md": f"{month_str}-%"}).fetchone()[0]
            
            pct = (present_days / MONTHLY_WORKING_DAYS) * 100
            pct = min(100.0, round(pct, 1))
            
            report.append({
                "employee_id": emp_id,
                "name": emp_name,
                "present_days": present_days,
                "total_days": MONTHLY_WORKING_DAYS,
                "percentage": f"{pct}%"
            })
        
    report.sort(key=lambda x: float(x["percentage"].replace("%", "")), reverse=True)
    return report


def export_monthly_to_excel(month_str: str, filepath: str):
    report_data = get_monthly_report(month_str)
    df = pd.DataFrame(report_data)
    df.columns = ["Employee ID", "Employee Name", "Present Days", "Monthly Working Days", "Attendance %"]
    df.to_excel(filepath, index=False)
    return filepath


def update_camera_status(camera_name: str, status: str, downtime_start: str = None, last_seen: str = None):
    with get_engine().begin() as conn:
        chk_query = text("SELECT camera_name FROM camera_status WHERE camera_name = :cam")
        row = conn.execute(chk_query, {"cam": camera_name}).fetchone()
        
        if row:
            upd = text("""
                UPDATE camera_status SET 
                    status = :st,
                    downtime_start = CASE WHEN :ds IS NOT NULL THEN :ds ELSE downtime_start END,
                    last_seen = CASE WHEN :ls IS NOT NULL THEN :ls ELSE last_seen END
                WHERE camera_name = :cam
            """)
            conn.execute(upd, {"cam": camera_name, "st": status, "ds": downtime_start, "ls": last_seen})
        else:
            ins = text("""
                INSERT INTO camera_status (camera_name, status, downtime_start, last_seen)
                VALUES (:cam, :st, :ds, :ls)
            """)
            conn.execute(ins, {"cam": camera_name, "st": status, "ds": downtime_start, "ls": last_seen})


def get_camera_status():
    with get_engine().connect() as conn:
        rows = conn.execute(text("SELECT camera_name, status, last_seen, downtime_start FROM camera_status")).fetchall()
        return [dict(r._mapping) for r in rows]


def log_unknown_detection(timestamp: str, camera_source: str, image_name: str):
    with get_engine().begin() as conn:
        conn.execute(
            text("INSERT INTO unknown_detections (timestamp, camera_source, image_name) VALUES (:ts, :cs, :im)"),
            {"ts": timestamp, "cs": camera_source, "im": image_name}
        )


def get_recent_unknowns(limit: int = 10):
    with get_engine().connect() as conn:
        query = text("SELECT timestamp, camera_source, image_name FROM unknown_detections ORDER BY timestamp DESC LIMIT :lim")
        rows = conn.execute(query, {"lim": limit}).fetchall()
        return [dict(r._mapping) for r in rows]
