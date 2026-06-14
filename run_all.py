import sys
import os
import threading
import time
import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attendance_db import initialise_database
from camera_processor import AttendanceProcessor
from dashboard.app import app
import logging as _log

def run_monitor(attendance_processor):
    print("[STEP 2] Starting entry / exit attendance cameras...")
    attendance_processor.start()

    print("\n" + "=" * 60)
    print("  MONITOR & DASHBOARD ARE RUNNING")
    print("=" * 60)
    print("  Attendance:  Camera 1 (Entry) + Camera 2 (Exit)")
    print("  Dashboard:   http://0.0.0.0:5000")
    print("  Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    while True:
        time.sleep(5)

def main():
    print("[STEP 1] Initialising databases...")
    initialise_database()

    attendance_processor = AttendanceProcessor()
    
    # Run the camera monitor in a background thread
    monitor_thread = threading.Thread(target=run_monitor, args=(attendance_processor,), daemon=True)
    monitor_thread.start()

    def handle_shutdown(sig, frame):
        print("\n\n[SYSTEM] Shutdown signal received. Stopping...")
        attendance_processor.stop()
        print("[SYSTEM] All systems stopped. Goodbye.\n")
        sys.exit(0)

    signal.signal(signal.SIGINT,  handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # Run the dashboard in the main thread
    _log.getLogger("werkzeug").setLevel(_log.ERROR)
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

if __name__ == "__main__":
    main()
