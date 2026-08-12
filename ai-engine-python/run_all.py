import os
import sys
import time
import subprocess

def main():
    print("=" * 60)
    print(" 🚀 Launching Multimodal E-Commerce Search Engine Stack")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Start FastAPI Backend Server
    print("\n[1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ...")
    api_cmd = [sys.executable, "-m", "uvicorn", "src.api:app", "--host", "127.0.0.1", "--port", "8000"]
    api_process = subprocess.Popen(api_cmd, cwd=base_dir)

    # Give backend 4 seconds to load CLIP & FAISS into memory
    print("Waiting 4 seconds for AI engine initialization...")
    time.sleep(4)

    # 2. Start Streamlit Frontend Server
    print("\n[2/2] Starting Streamlit Web UI on http://localhost:8501 ...")
    app_path = os.path.join(base_dir, "app.py")
    st_cmd = [sys.executable, "-m", "streamlit", "run", app_path]
    st_process = subprocess.Popen(st_cmd, cwd=base_dir)

    print("\n" + "=" * 60)
    print("  ✅ BOTH SERVERS RUNNING SUCCESSFULLY!")
    print("  • Backend API : http://127.0.0.1:8000")
    print("  • Frontend UI : http://localhost:8501")
    print("  Press CTRL+C in this terminal to shut down both servers.")
    print("=" * 60 + "\n")

    try:
        api_process.wait()
        st_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down both servers cleanly...")
        st_process.terminate()
        api_process.terminate()
        print("Shutdown complete!")

if __name__ == "__main__":
    main()
