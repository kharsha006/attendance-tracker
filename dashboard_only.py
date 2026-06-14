import sys
import os
import logging as _log

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the initialized Flask app
from dashboard.app import app

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  OFFICE ATTENDANCE DASHBOARD")
    print("=" * 60)
    print("  Running on:   http://0.0.0.0:5000")
    print("  To run in production: gunicorn dashboard_only:app")
    print("=" * 60 + "\n")
    
    _log.getLogger("werkzeug").setLevel(_log.ERROR)
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
