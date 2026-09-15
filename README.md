# Tarea 1, IIC3533 Computación de Alto Rendimiento (2026-2)

Bootstrapping paralelo para regresión lineal con `joblib`. Grupo de 3 personas, con experimentos en al menos 2 computadores distintos. La entrega es el viernes 11 de septiembre de 2026 a las 23:59.

## Estado actual

- [x] **(a) Generación de datos**, `generate_data.py`
- [x] **(b) Tres implementaciones**, `bs_auto.py`, `bs_sklearn.py`, `bs_numpy.py`, más `common.py` con utilidades compartidas
- [x] **(c) Correctitud y reproducibilidad**, `verify_correctness.py`
- [x] **(d) Cómo crea procesos el backend `multiprocessing` de joblib**
- [x] **(e) Oversubscription con `threadpoolctl`**, `oversubscription.py`
- [x] **(f) Tiempos T(p) para `p = 1..p_max`, 3 versiones**, `benchmark.py`
- [x] **(g) Speedup S(p) y eficiencia E(p)**, `speedup_efficiency.py`
- [x] **(h) Overhead T_o(p)**, `overhead.py`
- [x] **(i) Grid de procesos p y threads t con `p·t ≤ p_max`**, `grid_pt.py`
- [ ] **(j) Comparación entre los 2 computadores**
- [ ] Informe final en PDF

## Cómo generar los datos

```bash
python generate_data.py