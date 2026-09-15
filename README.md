# Tarea 1 — IIC3533 Computación de Alto Rendimiento (2026-2)

**Bootstrapping paralelo para regresión lineal con `joblib`**

- Trabajo grupal (3 personas)
- Experimentos en al menos 2 computadores distintos
- **Fecha de entrega:** viernes 25 de septiembre de 2026, 23:59

---

## Estado actual del proyecto

| Ítem | Descripción | Script(s) | Estado |
|:---:|---|---|:---:|
| (a) | Generación de datos sintéticos ($N=100\,000$, $k=300$, $B=48$) | `generate_data.py` | ✅ |
| (b) | Tres implementaciones del algoritmo | `bs_auto.py`, `bs_sklearn.py`, `bs_numpy.py`, `common.py` | ✅ |
| (c) | Correctitud y reproducibilidad | `verify_correctness.py` | ✅ |
| (d) | Análisis del backend multiprocessing de Joblib (memoria virtual, `numpy.memmap`) | — | ✅ |
| (e) | Estudio de oversubscription | `oversubscription.py`, `threadpoolctl` | ✅ |
| (f) | Tiempos de ejecución $T(p)$ para $p \in \{1,\dots,16\}$, 3 repeticiones | `benchmark.py` | ✅ |
| (g) | Speedup $S(p)$ y eficiencia $E(p)$ | `speedup_efficiency.py` | ✅ |
| (h) | Overhead paralelo $T_o(p)$ | `overhead.py` | ✅ |
| (i) | Grilla bidimensional $(p,t)$ con $p\cdot t \le p_{\max}$ | `grid_pt.py` | ✅ |
| (j) | Comparación entre dos computadores | `run_pc2.py` | ⬜ **pendiente** |
| — | Informe en PDF (LaTeX, `main.tex`) | — | ✅ |

---

## Parámetros del problema

| Parámetro | Valor |
|---|---|
| Observaciones ($N$) | 100 000 |
| Variables de entrada ($k$) | 300 (matriz $X$: $100\,000 \times 301$, incluye columna de unos para el intercepto) |
| Resamples bootstrap ($B$) | 48 |
| Tamaño de $X$ (float64) | ≈ 240.8 MB |
| Tamaño de $y$ | ≈ 800 KB |
| Semilla determinista base | `42` |

---

## Instalación y configuración del entorno

Se puede usar tanto `conda` como `venv` con **Python 3.13**.

### Opción 1: venv (recomendada si ya se tiene Python 3.13)

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install numpy scikit-learn joblib threadpoolctl matplotlib psutil pandas
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scikit-learn joblib threadpoolctl matplotlib psutil pandas
```

---

## Estructura del repositorio

```
├── .venv/                   # Entorno virtual de Python
├── results/                 # CSVs de mediciones y datos de escalamiento (PC 1)
├── results_pc2/             # Carpeta de salida para las mediciones del PC 2
├── X.npy                    # Matriz de diseño (100000, 301) - 240.8 MB
├── y.npy                    # Vector objetivo (100000,) - 800 KB
├── beta_true.npy            # Coeficientes verdaderos (301,)
├── common.py                # Funciones compartidas (carga de datos, cálculo de IC bootstrap)
├── generate_data.py         # Generador de datos sintéticos (ítem a)
├── bs_auto.py               # Bootstrap con BaggingRegressor de sklearn (ítem b)
├── bs_sklearn.py            # Bootstrap con joblib.Parallel + LinearRegression (ítem b)
├── bs_numpy.py              # Bootstrap con joblib.Parallel + Ecuaciones Normales (ítem b)
├── verify_correctness.py    # Verificación de reproducibilidad y correctitud numérica (ítem c)
├── oversubscription.py      # Inspección de threads y oversubscription BLAS (ítem e)
├── benchmark.py             # Barrido de p=1..p_max para las 3 versiones (ítem f)
├── speedup_efficiency.py    # Cálculo y gráficos de S(p) y E(p) (ítem g)
├── overhead.py              # Cálculo y análisis de To(p) (ítem h)
├── grid_pt.py               # Exploración de grilla (p, t) para bs_numpy (ítem i)
├── analyze_benchmark.py     # Script auxiliar de análisis de datos
├── run_pc2.py               # Script automatizado todo-en-uno para el Computador 2 (ítem j)
└── README.md
```

---

## Resumen de resultados

**Computador 1:** 16 núcleos lógicos / 8 físicos

### 1. Implementaciones y tiempos base ($p=1$, $t=1$, $B=48$)

| Script | Tiempo | Detalle |
|---|---:|---|
| `bs_numpy.py` | 14.165 s | Resuelve $(X^\top X)\hat{\beta} = X^\top y$ con `np.linalg.solve`, explotando BLAS nivel 3 (`dgemm`/`dsyrk`) |
| `bs_sklearn.py` | 66.570 s | Usa `LinearRegression` (resuelve por `lstsq` SVD/QR), con sobrecarga de validaciones y copias |
| `bs_auto.py` | 63.640 s | Usa `BaggingRegressor`, con sobrecarga de orquestación interna |

**Mejoras aplicadas:**
- Reemplazo de `np.linalg.inv` por `np.linalg.solve` (reducción drástica de tiempo y mayor estabilidad numérica)
- Control explícito de hilos BLAS con `threadpoolctl`
- Aislamiento de resamples mediante `SeedSequence.spawn`

### 2. Correctitud y reproducibilidad (ítem c)

- **Reproducibilidad bit a bit:** verificada (`True` para las tres versiones entre ejecuciones independientes)
- **Equivalencia numérica** (`bs_numpy` vs `bs_sklearn`):
  - Diferencia absoluta máxima: ≈ $5 \times 10^{-14}$
  - Diferencia absoluta promedio: ≈ $4 \times 10^{-15}$
  - La discrepancia responde únicamente a la no asociatividad del punto flotante IEEE 754 por el diferente orden de acumulación de operaciones.
- **Cobertura estadística de los IC 95% sobre $\beta^*$:**

  | Implementación | Cobertura |
  |---|---|
  | `bs_numpy` | 279 / 301 coeficientes |
  | `bs_sklearn` | 279 / 301 coeficientes |
  | `bs_auto` | 281 / 301 coeficientes |

### 3. Gestión de memoria de Joblib (ítem d)

Dado que $X$ pesa ≈ 240.8 MB, serializarlo por IPC (pickle) para $p$ procesos requeriría $p \times 240.8\text{ MB}$ de RAM adicional. Joblib utiliza `numpy.memmap` volcando los datos a `/dev/shm` con `mmap_mode='r'`, de modo que todos los procesos leen directamente las mismas páginas físicas de RAM sin duplicación.

### 4. Oversubscription (ítem e)

OpenBLAS configura 16 threads por defecto.

- En $p=1$: aumentar de $t=1$ (16.685 s) a $t=16$ (12.959 s) mejora el desempeño monoproceso.
- En $p=16$: aumentar $t$ produce saturación de CPU por arbitraje del sistema operativo (system time / context switching).
- **Conclusión:** para paralelismo de procesos múltiples, fijar $t=1$ con `threadpoolctl` es la estrategia estándar.

### 5. Tiempos, Speedup, Eficiencia y Overhead ($p = 1 \dots 16$)

| $p$ | $T_{numpy}$ (s) | $S_{numpy}$ | $E_{numpy}$ | $T_{sklearn}$ (s) | $T_{auto}$ (s) |
|---:|---:|---:|---:|---:|---:|
| 1  | 14.165 | 1.00 | 1.00 | 66.570 | 63.640 |
| 2  | 6.944  | 2.04 | 1.02 | 44.124 | 41.625 |
| 4  | 4.829  | 2.93 | 0.73 | 38.911 | 39.823 |
| 6  | 4.296  | 3.30 | 0.55 | 38.647 | 41.688 |
| 8  | 3.704  | 3.82 | 0.48 | 37.778 | 40.237 |
| 10 | 3.361  | 4.21 | 0.42 | 37.013 | 37.686 |
| 12 | 3.443  | 4.11 | 0.34 | 37.776 | 38.273 |
| 14 | 3.563  | 3.98 | 0.28 | 37.905 | 39.636 |
| 16 | 3.484  | 4.06 | 0.25 | 37.606 | 39.827 |

- **Speedup superlineal en $p=2$:** explicado por la jerarquía de memoria (mayor tasa de aciertos / cache hits en cachés locales L1/L2 al particionar la carga de trabajo).
- **Comportamiento de $T_o(p)$:** crece con $p$ debido a contención en el bus de memoria RAM, barreras de sincronización y desbalance de carga estático cuando $B=48$ no es divisible por $p$ (e.g., escalones en $p = 7, 9, 11$).

### 6. Grilla $(p,t)$ con $p \cdot t \le 16$ (ítem i)

**Mejor configuración observada:** $p=8$, $t=2$ → **3.196 s**

**Justificación física:** la máquina cuenta con 8 núcleos físicos y 16 hilos lógicos (SMT / Hyper-Threading). Configurar $p=8$ asigna exactamente un proceso por núcleo físico (evitando sobrecarga de 16 intérpretes Python), mientras que $t=2$ permite a OpenBLAS aprovechar el segundo hilo lógico para solapar instrucciones aritméticas FMA y accesos a memoria en el cálculo de $X^\top X$.

---

## Instrucciones para el Computador 2 (ítem j)

Para completar la comparación de hardware:

1. Clonar el repositorio y asegurar que `X.npy`, `y.npy` y `beta_true.npy` estén en la raíz (para garantizar que se resuelva exactamente el mismo problema).
2. Crear y activar el entorno (`venv`).
3. Conectar el equipo a la corriente alterna y cerrar programas en segundo plano.
4. Ejecutar el script automatizado:
   ```bash
   python run_pc2.py
   ```

El script generará la carpeta `results_pc2/` con especificaciones técnicas, tablas CSV y figuras (`metricas_rendimiento_pc2.png` y `heatmap_pt_pc2.png`). Comprimir dicha carpeta y traspasarla para cerrar la Sección 11 del informe.
