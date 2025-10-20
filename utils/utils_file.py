import subprocess
import threading
from datetime import datetime


print_lock = threading.Lock()

def log_message(msg):
    with print_lock:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {msg}", flush=True)

        
def run_front_bash_script(script_path, *args):
    """Lance un script bash pour la mesure de puissance."""
    try:
        process = subprocess.Popen([script_path, *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log_message(f"Script {script_path} lancé (PID: {process.pid})")
        for line in process.stdout:
            print(line, end="")
        process.wait()
        return process
    except Exception as e:
        log_message(f"Erreur lancement script {script_path}: {e}")
        return None



def run_back_bash_script(script_path, *args):
    """Lance un script bash pour la mesure de puissance."""
    try:
        process = subprocess.Popen([script_path, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        log_message(f"Script {script_path} lancé (PID: {process.pid})")
        return process
    except Exception as e:
        log_message(f"Erreur lancement script {script_path}: {e}")
        return None
