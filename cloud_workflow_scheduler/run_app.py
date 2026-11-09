#!/usr/bin/env python3
"""
run_app.py
==========
Concurrent launcher for the Q-Learning Scheduler Dashboard:
- FastAPI Backend (Uvicorn) on http://localhost:8000
- React Frontend (Vite) on http://localhost:5173

Usage:
    python run_app.py
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT / "frontend"


def check_node_installed():
    try:
        subprocess.run(["node", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        subprocess.run(["npm", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: Node.js and npm must be installed to run the frontend dashboard.")
        sys.exit(1)


def ensure_frontend_deps():
    node_modules = FRONTEND_DIR / "node_modules"
    if not node_modules.exists():
        print("📦 Installing frontend npm dependencies (this may take a minute on first run)...")
        subprocess.run(["npm", "install"], cwd=str(FRONTEND_DIR), check=True)


def main():
    print("=" * 70)
    print("🚀 Starting Q-Learning Cloud Workflow Scheduler Full-Stack App")
    print("=" * 70)

    check_node_installed()
    ensure_frontend_deps()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    # 1. Start FastAPI backend
    print("📡 Launching FastAPI backend server on http://localhost:8000 ...")
    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api.server:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--reload",
    ]
    backend_proc = subprocess.Popen(
        backend_cmd,
        cwd=str(ROOT),
        env=env,
    )

    # Wait 1.5 seconds for backend to initialize
    time.sleep(1.5)

    # 2. Start Vite React frontend
    print("💻 Launching Vite React frontend on http://localhost:5173 ...")
    frontend_cmd = ["npm", "run", "dev", "--", "--host"]
    frontend_proc = subprocess.Popen(
        frontend_cmd,
        cwd=str(FRONTEND_DIR),
    )

    print("\n" + "=" * 70)
    print("✨ Both servers are running!")
    print("👉 Frontend Dashboard : http://localhost:5173")
    print("👉 Backend API Docs   : http://localhost:8000/docs")
    print("👉 WebSocket Stream   : ws://localhost:8000/ws/live-stream")
    print("=" * 70)
    print("Press Ctrl+C to terminate both servers...\n")

    def shutdown(sig, frame):
        print("\n🛑 Shutting down servers...")
        try:
            frontend_proc.terminate()
            backend_proc.terminate()
            frontend_proc.wait(timeout=3)
            backend_proc.wait(timeout=3)
        except Exception:
            frontend_proc.kill()
            backend_proc.kill()
        print("✅ Shutdown complete. Goodbye!")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Monitor processes
    while True:
        if backend_proc.poll() is not None:
            print("⚠️ Backend server terminated unexpectedly.")
            frontend_proc.terminate()
            sys.exit(1)
        if frontend_proc.poll() is not None:
            print("⚠️ Frontend server terminated unexpectedly.")
            backend_proc.terminate()
            sys.exit(1)
        time.sleep(1)


if __name__ == "__main__":
    main()

