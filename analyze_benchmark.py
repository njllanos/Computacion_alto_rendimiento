import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

IDEAL_STYLE = {"linestyle": "--", "color": "black", "alpha": 0.6}


def load_csv(path):
    return pd.read_csv(path)


def compute_summary(df):
    # Mediana por version y p
    summary = (
        df.groupby(["version", "p"], as_index=False)["segundos"]
        .median()
        .rename(columns={"segundos": "T_p"})
        .sort_values(["version", "p"])
    )

    # T(1) por version
    t1 = summary[summary["p"] == 1][["version", "T_p"]].rename(columns={"T_p": "T_1"})
    summary = summary.merge(t1, on="version", how="left")

    summary["S_p"] = summary["T_1"] / summary["T_p"]
    summary["E_p"] = summary["S_p"] / summary["p"]
    summary["T_o_p"] = summary["p"] * summary["T_p"] - summary["T_1"]

    return summary


def print_tables(summary):
    for version in summary["version"].unique():
        sub = summary[summary["version"] == version].copy()
        print(f"\n=== {version} ===")
        print(sub[["p", "T_p", "S_p", "E_p", "T_o_p"]].to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def plot_metric(summary, metric, ylabel, title, outpath):
    plt.figure(figsize=(8, 5))

    for version in summary["version"].unique():
        sub = summary[summary["version"] == version]
        plt.plot(sub["p"], sub[metric], marker="o", label=version)

    # Curva ideal cuando aplica
    if metric == "S_p":
        p = sorted(summary["p"].unique())
        plt.plot(p, p, label="ideal", **IDEAL_STYLE)
    elif metric == "E_p":
        p = sorted(summary["p"].unique())
        plt.plot(p, [1.0] * len(p), label="ideal", **IDEAL_STYLE)

    plt.xlabel("p")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Archivo CSV generado por benchmark.py")
    parser.add_argument("--outdir", default="results/plots", help="Directorio de salida")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_csv(csv_path)
    summary = compute_summary(df)

    print_tables(summary)
    summary.to_csv(outdir / "summary.csv", index=False)

    plot_metric(summary, "T_p", "T(p) [s]", "Tiempo de ejecucion T(p)", outdir / "T_p.png")
    plot_metric(summary, "S_p", "S(p)", "Speedup S(p)", outdir / "S_p.png")
    plot_metric(summary, "E_p", "E(p)", "Eficiencia E(p)", outdir / "E_p.png")
    plot_metric(summary, "T_o_p", "T_o(p) [s]", "Overhead T_o(p)", outdir / "T_o_p.png")

    print(f"\nResumen guardado en: {outdir / 'summary.csv'}")
    print(f"Graficas guardadas en: {outdir}")


if __name__ == "__main__":
    main()