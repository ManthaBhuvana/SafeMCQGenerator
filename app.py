import os
import sys
import subprocess
import threading

def run_streamlit():
    # Set environment variables to disable problematic behaviors
    os.environ["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))
    os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"
    os.environ["STREAMLIT_SERVER_ENABLE_CORS"] = "false"
    os.environ["STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"] = "false"
    
    # Run the actual app through subprocess to isolate it
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "mcq_app.py",
        "--server.runOnSave=false",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false"
    ])

if __name__ == "__main__":
    # Start in a thread to avoid blocking
    thread = threading.Thread(target=run_streamlit)
    thread.daemon = True
    thread.start()
    
    # Simple message to indicate it's running
    print("MCQ Generator is starting...")
    
    # Keep the main process alive
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Shutting down...")
        sys.exit(0)
