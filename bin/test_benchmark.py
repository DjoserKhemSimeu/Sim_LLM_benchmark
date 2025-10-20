import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from ollama import chat, ChatResponse
import matplotlib.pyplot as plt
import subprocess
import numpy as np
from matplotlib.ticker import MaxNLocator

T = 60.0  # temps du benchmark
lamb = 0.2  # nombre de requête moyen par pas de temps
Max = 128  # nb max requête par utilisateur

# Verrou pour les prints
print_lock = threading.Lock()

def run_bash_script(script_path, *args):
    try:
        process = subprocess.Popen([script_path, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with print_lock:
            print(f"Script lancé en arrière-plan avec le PID : {process.pid}")
        return process
    except Exception as e:
        with print_lock:
            print(f"Une erreur s'est produite : {e}")
        return None

def simulate_user(user_id: int, model='mistral:7b', delta_t_collector=None):
    times = []
    delta_ts = []
    start = time.time()
    i = 0
    while time.time() < start + T and i < Max:
        delta_t = np.random.exponential(1/lamb)
        delta_ts.append(delta_t)
        time.sleep(delta_t)
        with print_lock:
            print(f"[User {user_id}] Delta compute: {delta_t:.3f}s")
        if time.time() < start + T:
            start_req = time.time()
            try:
                response: ChatResponse = chat(model=model, messages=[
                    {'role': 'user', 'content': f'Hello from user {user_id}, request {i}'}
                ])
                elapsed = time.time() - start_req
                times.append(elapsed)
                with print_lock:
                    print(f"[User {user_id}] Request {i} latency: {elapsed:.3f}s")
                i += 1
            except Exception as e:
                with print_lock:
                    print(f"[User {user_id}] Error on request {i}: {e}")
    if delta_t_collector is not None:
        delta_t_collector[user_id] = delta_ts
    return times

def benchmark(users_list=[2]):
    results = {}
    delta_t_data = {}
    for n_users in users_list:
        with print_lock:
            print(f"\nRunning benchmark with {n_users} concurrent users...")
        # Démarrer la mesure de puissance
        start_process = run_bash_script('./measure/script_start_tx.sh', str(n_users))
        if start_process is None:
            with print_lock:
                print("Erreur au démarrage de la mesure de puissance.")
            continue
        all_times = []
        delta_t_collector = {}
        with ThreadPoolExecutor(max_workers=n_users) as executor:
            futures = [executor.submit(simulate_user, user_id=i, delta_t_collector=delta_t_collector) for i in range(n_users)]
            for future in as_completed(futures):
                all_times.extend(future.result())
        # Arrêter la mesure de puissance
        stop_process = run_bash_script('./measure/script_stop_tx.sh')
        if stop_process is None:
            with print_lock:
                print("Erreur à l'arrêt de la mesure de puissance.")
        results[n_users] = all_times
        delta_t_data[n_users] = [dt for user in delta_t_collector for dt in delta_t_collector[user]]
        with print_lock:
            print(f"Completed {n_users} users, total requests: {len(all_times)}")
        time.sleep(5)
    return results, delta_t_data

def plot_results(results, delta_t_data):
    users = sorted(results.keys())
    latencies = [results[u] for u in users]
    n_requests = [len(results[u]) for u in users]
    avg_requests_per_user = [np.mean([len(results[u])/u for _ in range(u)]) for u in users]

    # --- Graphique principal ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Boxplot des latences
    bp = ax1.boxplot(latencies, positions=users, widths=0.6, patch_artist=True, showfliers=False)
    ax1.set_xticks(users)
    ax1.set_xticklabels(users)
    ax1.set_ylabel("Latency per request (s)", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_title("LLM Latency and Requests per User")
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Axe Y secondaire pour le nombre de requêtes (même axe x)
    ax1b = ax1.twinx()
    ax1b.plot(users, avg_requests_per_user, 'r-', marker='o', label='Avg requests per user')
    ax1b.set_ylabel("Avg requests per user", color='tab:red')
    ax1b.tick_params(axis='y', labelcolor='tab:red')
    ax1b.yaxis.set_major_locator(MaxNLocator(integer=True))

    # --- Graphique des delta_t ---
    colors = plt.cm.viridis(np.linspace(0, 1, len(users)))
    for i, u in enumerate(users):
        ax2.hist(delta_t_data[u], bins=30, alpha=0.5, density=True,
                 label=f'{u} users', color=colors[i], edgecolor='black')
    ax2.set_xlabel("Delta_t (s)")
    ax2.set_ylabel("Density")
    ax2.set_title("Distribution of delta_t per number of users")
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("./measure/data/results_bench.png")

if __name__ == "__main__":
    benchmark_results, delta_t_data = benchmark([2, 4, 8])
    plot_results(benchmark_results, delta_t_data)

