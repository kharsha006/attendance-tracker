import sys
import os
import time
import signal
from tz_utils import now_ist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attendance_db import initialise_database
from camera_processor import AttendanceProcessor
from config import EMBEDDINGS_FILE
from employee_db import EmployeeDB

def check_prerequisites():
    print("\n" + "=" * 60)
    print("  OFFICE ATTENDANCE MONITOR (Phase 3)")
    print("=" * 60)

    db_ok = False
    
    try:
        emp_db = EmployeeDB()
        count = emp_db.count()
        if count > 0:
            print(f"\n[OK] Employee database found ({count} employee(s) enrolled).")
            db_ok = True
        else:
            print("\n[WARNING] employees database exists but has 0 employees enrolled.")
    except Exception as e:
        pass

    if not db_ok:
        # Fallback: check legacy pkl
        if os.path.exists(EMBEDDINGS_FILE):
            print(f"\n[WARNING] employees database empty/missing — using legacy embeddings.pkl.")
            db_ok = True
        else:
            print("\n[ERROR] No employee database found.")
            print("  Primary: data/employees.db (run: python enroll_employees.py)")
            print("  Legacy : enrollment/embeddings.pkl")
            sys.exit(1)

    print("[OK] Prerequisites met.\n")


def main():
    check_prerequisites()

    print("[STEP 1] Initialising databases...")
    initialise_database()

    print("[STEP 2] Starting entry / exit attendance cameras...")
    attendance_processor = AttendanceProcessor()
    attendance_processor.start()

    print("\n" + "=" * 60)
    print("  MONITOR IS RUNNING")
    print("=" * 60)
    print("  Attendance is currently tracking via your camera(s).")
    print("  Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    print("=" * 60 + "\n")

    def handle_shutdown(sig, frame):
        print("\n\n[SYSTEM] Shutdown signal received. Stopping...")
        attendance_processor.stop()
        print("[SYSTEM] All systems stopped. Goodbye.\n")
        sys.exit(0)

    signal.signal(signal.SIGINT,  handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    while True:
        time.sleep(5)
        now = now_ist()
        if now.second < 5 and now.minute % 5 == 0:
            att_ok   = attendance_processor.is_running()
            status   = "OK" if att_ok else "WARNING"
            print(f"[HEARTBEAT] {now.strftime('%H:%M')} — "
                  f"Attendance={att_ok}  [{status}]")


if __name__ == "__main__":
    main()
