import os
import platform
import time
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits, threadpool_info
from joblib import Parallel, delayed

from bs_auto import run_bs_auto
from bs_sklearn import run_bs_sklearn
from bs_numpy import run_bs_numpy

# 0. Crear directorio de resultados
OUTPUT_DIR = "results_pc2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Registro de Hardware
P_MAX = os.cpu_count() or psutil.cpu_count(logical=True)
p_fisicos = psutil.cpu_count(logical=False)

specs_path = os.path.join(OUTPUT_DIR, "specs_pc2.txt")
with open(specs_path, "w") as f:
    f.write(f"SO: {platform.system()} {platform.release()}\n")
    f.write(f"Procesador: {platform.processor()}\n")
    f.write(f"Cores Fisicos: {p_fisicos}\n")
    f.write(f"Cores Logicos (p_max): {P_MAX}\n")
    f.write(f"RAM Total: {psutil.virtual_memory().total / (1024**3):.2f} GB\n")
    f.write(f"BLAS Info: {threadpool_info()}\n")

print(f"=== PC 2: Cores Físicos: {p_fisicos} | Cores Lógicos: {P_MAX} ===")

# 2. Carga de datos idénticos
print("Cargando matrices de datos...")
X = np.load("X.npy")
y = np.load("y.npy")

REPETICIONES = 3

# 3. Experimentos (f), (g), (h): p = 1 hasta p_max
print(f"\n>>> [1/2] Barrido p = 1..{P_MAX} con {REPETICIONES} repeticiones...")
res_f = []

for p in range(1, P_MAX + 1):
    t_np_runs, t_sk_runs, t_au_runs = [], [], []
    for rep in range(REPETICIONES):
        print(f"  p = {p}/{P_MAX} | Repeticion {rep + 1}/{REPETICIONES}...")
        t_np, _, _, _ = run_bs_numpy(X, y, p=p)
        t_sk, _, _, _ = run_bs_sklearn(X, y, p=p)
        t_au, _, _, _ = run_bs_auto(X, y, p=p)
        
        t_np_runs.append(t_np)
        t_sk_runs.append(t_sk)
        t_au_runs.append(t_au)
        
    res_f.append({
        "p": p,
        "t_numpy": np.mean(t_np_runs),
        "t_sklearn": np.mean(t_sk_runs),
        "t_auto": np.mean(t_au_runs)
    })

df_f = pd.DataFrame(res_f)

# Cálculo de Speedup, Eficiencia y Overhead formal (Clase 07)
for v in ["numpy", "sklearn", "auto"]:
    t1 = df_f.loc[df_f["p"] == 1, f"t_{v}"].values[0]
    df_f[f"S_{v}"] = t1 / df_f[f"t_{v}"]
    df_f[f"E_{v}"] = df_f[f"S_{v}"] / df_f["p"]
    df_f[f"To_{v}"] = df_f["p"] * df_f[f"t_{v}"] - t1

csv_scaling_path = os.path.join(OUTPUT_DIR, "benchmark_scaling_p_pc2.csv")
df_f.to_csv(csv_scaling_path, index=False)

# Generar Figuras de Escalamiento
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# T(p)
axs[0, 0].plot(df_f["p"], df_f["t_auto"], 'o-', label="bs_auto")
axs[0, 0].plot(df_f["p"], df_f["t_sklearn"], 's-', label="bs_sklearn")
axs[0, 0].plot(df_f["p"], df_f["t_numpy"], '^-', label="bs_numpy")
axs[0, 0].set_title("Tiempo de Ejecución T(p)")
axs[0, 0].set_xlabel("Procesos (p)")
axs[0, 0].set_ylabel("Tiempo (s)")
axs[0, 0].grid(True, linestyle="--", alpha=0.6)
axs[0, 0].legend()

# S(p)
axs[0, 1].plot(df_f["p"], df_f["p"], 'k--', label="Ideal")
axs[0, 1].plot(df_f["p"], df_f["S_auto"], 'o-', label="bs_auto")
axs[0, 1].plot(df_f["p"], df_f["S_sklearn"], 's-', label="bs_sklearn")
axs[0, 1].plot(df_f["p"], df_f["S_numpy"], '^-', label="bs_numpy")
axs[0, 1].set_title("Speedup S(p)")
axs[0, 1].set_xlabel("Procesos (p)")
axs[0, 1].set_ylabel("Speedup")
axs[0, 1].grid(True, linestyle="--", alpha=0.6)
axs[0, 1].legend()

# E(p)
axs[1, 0].axhline(1.0, color='k', linestyle='--', label="Ideal")
axs[1, 0].plot(df_f["p"], df_f["E_auto"], 'o-', label="bs_auto")
axs[1, 0].plot(df_f["p"], df_f["E_sklearn"], 's-', label="bs_sklearn")
axs[1, 0].plot(df_f["p"], df_f["E_numpy"], '^-', label="bs_numpy")
axs[1, 0].set_title("Eficiencia E(p)")
axs[1, 0].set_xlabel("Procesos (p)")
axs[1, 0].set_ylabel("Eficiencia")
axs[1, 0].set_ylim(0, 1.15)
axs[1, 0].grid(True, linestyle="--", alpha=0.6)
axs[1, 0].legend()

# To(p)
axs[1, 1].plot(df_f["p"], df_f["To_auto"], 'o-', label="bs_auto")
axs[1, 1].plot(df_f["p"], df_f["To_sklearn"], 's-', label="bs_sklearn")
axs[1, 1].plot(df_f["p"], df_f["To_numpy"], '^-', label="bs_numpy")
axs[1, 1].set_title("Overhead Paralelo T_o(p)")
axs[1, 1].set_xlabel("Procesos (p)")
axs[1, 1].set_ylabel("Overhead (s)")
axs[1, 1].grid(True, linestyle="--", alpha=0.6)
axs[1, 1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "metricas_rendimiento_pc2.png"), dpi=300)
plt.close()

# 4. Experimento (i): Grilla (p, t) con p * t <= p_max
print(f"\n>>> [2/2] Grilla (p, t) con p * t <= {P_MAX}...")
res_i = []

def worker_pt(s, X_mat, y_vec, N_obs, t_limit):
    with threadpool_limits(limits=t_limit):
        rng = np.random.default_rng(s)
        idx = rng.integers(0, N_obs, size=N_obs)
        X_b = X_mat[idx]
        y_b = y_vec[idx]
        return np.linalg.solve(X_b.T @ X_b, X_b.T @ y_b)

matrix_times = np.full((P_MAX, P_MAX), np.nan)

for p in range(1, P_MAX + 1):
    for t in range(1, P_MAX + 1):
        if p * t <= P_MAX:
            print(f"  Probando (p={p}, t={t})...")
            ss = np.random.SeedSequence(42)
            t0 = time.perf_counter()
            Parallel(n_jobs=p, backend="loky")(
                delayed(worker_pt)(s, X, y, X.shape[0], t) for s in ss.spawn(48)
            )
            t_total = time.perf_counter() - t0
            res_i.append({"p": p, "t": t, "tiempo": t_total})
            matrix_times[p - 1, t - 1] = t_total

df_i = pd.DataFrame(res_i)
csv_grid_path = os.path.join(OUTPUT_DIR, "grid_p_t_pc2.csv")
df_i.to_csv(csv_grid_path, index=False)

# Generar Heatmap equivalente
plt.figure(figsize=(10, 8))
masked_matrix = np.ma.masked_invalid(matrix_times)
cmap = plt.cm.viridis
cmap.set_bad(color='white')

im = plt.imshow(masked_matrix, cmap=cmap, origin='lower')
cbar = plt.colorbar(im)
cbar.set_label("Tiempo (s)")

for i in range(P_MAX):
    for j in range(P_MAX):
        val = matrix_times[i, j]
        if not np.isnan(val):
            plt.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val < np.nanmean(matrix_times) else "black", fontsize=8)

plt.xticks(ticks=range(P_MAX), labels=range(1, P_MAX + 1))
plt.yticks(ticks=range(P_MAX), labels=range(1, P_MAX + 1))
plt.xlabel("t (threads internos)")
plt.ylabel("p (procesos)")
plt.title(f"Tiempos de ejecución por combinación (p, t) - bs_numpy (PC 2)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "heatmap_pt_pc2.png"), dpi=300)
plt.close()

print(f"\n Proceso completado exitosamente. Todos los archivos están en '{OUTPUT_DIR}/'.")