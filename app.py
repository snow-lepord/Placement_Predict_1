"""
CampusPlacementIQ · Application Launcher
Launches the web intelligence dashboard from the project root.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DASHBOARD_DIR = os.path.join(PROJECT_ROOT, "dashboard")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, DASHBOARD_DIR)

from dashboard.app import app

if __name__ == "__main__":
    print("=" * 65)
    print("🚀 Starting CampusPlacementIQ ML Intelligence Platform")
    print("🌐 Dashboard URL: http://127.0.0.1:5001")
    print("=" * 65)
    app.run(debug=True, host="127.0.0.1", port=5001)
