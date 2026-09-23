"""
Genera 4 graficos comparativos entre PC 1 y PC 2 para el item (j).
Guarda en results/plots/comparacion_*.png (no sobreescribe graficos existentes).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Datos PC 1 ---
df_pc1 = pd.read_csv("results/plots/summary.csv")

# --- Datos PC 2 ---
df_pc2_raw = pd.read_csv("results/timings.csv")
df_pc2 = df_pc2_raw[["version", "p", "elapsed_seconds"]].copy()
df_pc2.rename(columns={"elapsed_seconds": "T_p"}, inplace=True)

for v in ["bs_numpy", "bs_sklearn", "bs_auto"]:
    t1 = df_pc2.loc[(df_pc2["version"] == v) & (df_pc2["p"] == 1), "T_p"].values[0]
    df_pc2.loc[df_pc2["version"] == v, "T_1"] = t1
    df_pc2.loc[df_pc2["version"] == v, "S_p"] = t1 / df_pc2.loc[df_pc2["version"] == v, "T_p"]
    df_pc2.loc[df_pc2["version"] == v, "E_p"] = df_pc2.loc[df_pc2["version"] == v, "S_p"] / df_pc2.loc[df_pc2["version"] == v, "p"]
    df_pc2.loc[df_pc2["version"] == v, "To_p"] = df_pc2.loc[df_pc2["version"] == v, "p"] * df_pc2.loc[df_pc2["version"] == v, "T_p"] - t1

P_MAX_PC2 = 4
df_pc1_sub = df_pc1[df_pc1["p"] <= P_MAX_PC2]

versiones = ["bs_numpy", "bs_sklearn", "bs_auto"]
colores   = {"bs_numpy": "#1f77b4", "bs_sklearn": "#ff7f0e", "bs_auto": "#2ca02c"}
p_ideal   = np.arange(1, P_MAX_PC2 + 1)


def base_axes(ax, titulo, xlabel, ylabel):
    ax.set_title(titulo, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(1, P_MAX_PC2 + 1))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)


# --- T(p) ---
fig, ax = plt.subplots(figsize=(7, 5))
for v in versiones:
    c = colores[v]
    d1 = df_pc1_sub[df_pc1_sub["version"] == v]
    d2 = df_pc2[df_pc2["version"] == v]
    ax.plot(d1["p"], d1["T_p"], "o-",  color=c, label=f"{v} — PC 1")
    ax.plot(d2["p"], d2["T_p"], "s--", color=c, label=f"{v} — PC 2", alpha=0.75)
base_axes(ax, "Tiempo de ejecución T(p)", "Procesos (p)", "Tiempo (s)")
plt.tight_layout()
plt.savefig("results/plots/comparacion_T_p.png", dpi=150)
plt.close()
print("Guardado: results/plots/comparacion_T_p.png")

# --- S(p) ---
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(p_ideal, p_ideal, "k--", label="Ideal", linewidth=1.2)
for v in versiones:
    c = colores[v]
    d1 = df_pc1_sub[df_pc1_sub["version"] == v]
    d2 = df_pc2[df_pc2["version"] == v]
    ax.plot(d1["p"], d1["S_p"], "o-",  color=c, label=f"{v} — PC 1")
    ax.plot(d2["p"], d2["S_p"], "s--", color=c, label=f"{v} — PC 2", alpha=0.75)
base_axes(ax, "Speedup S(p)", "Procesos (p)", "S(p)")
plt.tight_layout()
plt.savefig("results/plots/comparacion_S_p.png", dpi=150)
plt.close()
print("Guardado: results/plots/comparacion_S_p.png")

# --- E(p) ---
fig, ax = plt.subplots(figsize=(7, 5))
ax.axhline(1.0, color="k", linestyle="--", label="Ideal", linewidth=1.2)
for v in versiones:
    c = colores[v]
    d1 = df_pc1_sub[df_pc1_sub["version"] == v]
    d2 = df_pc2[df_pc2["version"] == v]
    ax.plot(d1["p"], d1["E_p"], "o-",  color=c, label=f"{v} — PC 1")
    ax.plot(d2["p"], d2["E_p"], "s--", color=c, label=f"{v} — PC 2", alpha=0.75)
base_axes(ax, "Eficiencia E(p)", "Procesos (p)", "E(p)")
ax.set_ylim(0, 1.2)
plt.tight_layout()
plt.savefig("results/plots/comparacion_E_p.png", dpi=150)
plt.close()
print("Guardado: results/plots/comparacion_E_p.png")

# --- To(p) ---
fig, ax = plt.subplots(figsize=(7, 5))
for v in versiones:
    c = colores[v]
    d1 = df_pc1_sub[df_pc1_sub["version"] == v]
    d2 = df_pc2[df_pc2["version"] == v]
    ax.plot(d1["p"], d1["T_o_p"], "o-",  color=c, label=f"{v} — PC 1")
    ax.plot(d2["p"], d2["To_p"],  "s--", color=c, label=f"{v} — PC 2", alpha=0.75)
base_axes(ax, "Overhead Tₒ(p)", "Procesos (p)", "Tₒ(p) (s)")
plt.tight_layout()
plt.savefig("results/plots/comparacion_To_p.png", dpi=150)
plt.close()
print("Guardado: results/plots/comparacion_To_p.png")
