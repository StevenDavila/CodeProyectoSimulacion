import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import CubicSpline, RectBivariateSpline

# =============================================================
# 1. CONFIGURACIÓN DEL PROBLEMA
# =============================================================
# Tamaño de la malla (nx puntos en x, ny puntos en y)
nx, ny = 80, 8
n = nx * ny  # número total de nodos

# (Opcional) Dimensiones físicas del dominio
Lx = 1.0  # largo del canal
Ly = 1.0  # alto del canal
x_phys = np.linspace(0.0, Lx, nx)
y_phys = np.linspace(0.0, Ly, ny)

# Condiciones de frontera (v=1 en pared izquierda, v=0 en las demás)
left_wall = 1.0
right_wall = top_wall = bottom_wall = 0.0

# =============================================================
# 2. CONSTRUCCIÓN DEL SISTEMA LINEAL (MATRIZ A Y VECTOR b)
# =============================================================
def build_system():
    """Construye la matriz dispersa A y el vector b del sistema Av = b."""
    A = lil_matrix((n, n), dtype=float)
    b = np.zeros(n, dtype=float)

    for i in range(nx):         # barrido en x
        for j in range(ny):     # barrido en y
            idx = i * ny + j    # índice lineal (orden x-fast / y-slow)

            # Término diagonal principal
            A[idx, idx] = -8.0  # -8 v_{i,j}

            # Vecinos en x (i±1, j)
            if i < nx - 1:
                A[idx, (i+1)*ny + j] = 1.0   # v_{i+1,j}
            if i > 0:
                A[idx, (i-1)*ny + j] = 3.0   # 3 v_{i-1,j}
            else:  # i == 0 -> frontera izquierda conocida
                b[idx] -= 3.0 * left_wall

            # Vecinos en y (i, j±1)
            if j < ny - 1:
                A[idx, i*ny + (j+1)] = 1.0   # v_{i,j+1}
            if j > 0:
                A[idx, i*ny + (j-1)] = 3.0   # 3 v_{i,j-1}
            else:  # j == 0 -> frontera inferior conocida
                b[idx] -= 3.0 * bottom_wall

            # Nota: fronteras top/right ya están en la ecuación base (-8*v + ... = 0)
            # porque sus valores son 0 -> no se suma nada a b.

    return csr_matrix(A), b

# Construir sistema
A, b = build_system()

# =============================================================
# 3. MÉTODO GAUSS–SEIDEL (ITERATIVO)
# =============================================================
def gauss_seidel(A, b, max_iter=100, tol=1e-4, x0=None, verbose=True):
    """Resuelve Av=b con Gauss–Seidel sobre matriz dispersa CSR.

    Parameters
    ----------
    A : csr_matrix
        Matriz del sistema.
    b : array_like
        Vector de términos independientes.
    max_iter : int
        Máximo de iteraciones.
    tol : float
        Norma L2 del residuo para detener.
    x0 : array_like or None
        Estimación inicial; si None se usa ceros.
    verbose : bool
        Imprime mensaje de convergencia.

    Returns
    -------
    v : ndarray (nx, ny)
        Solución reacomodada en malla.
    residuos : list[float]
        Norma del residuo en cada iteración.
    iters : int
        Iteraciones ejecutadas.
    """
    n_unknowns = b.size
    if x0 is None:
        v = np.zeros(n_unknowns, dtype=float)
    else:
        v = np.asarray(x0, dtype=float).copy()

    residuos = []
    diag = A.diagonal()  # guardar diagonal para acceso rápido

    for k in range(max_iter):
        # barrido Gauss–Seidel
        for i in range(n_unknowns):
            row_start = A.indptr[i]
            row_end = A.indptr[i+1]
            cols = A.indices[row_start:row_end]
            vals = A.data[row_start:row_end]

            # suma A[i,j]*v[j] excluyendo la diagonal
            mask = (cols != i)
            sum_ax = np.dot(vals[mask], v[cols[mask]])

            v[i] = (b[i] - sum_ax) / diag[i]

        # residuo
        r = A.dot(v) - b
        res = np.linalg.norm(r)
        residuos.append(res)

        if res < tol:
            if verbose:
                print(f"Convergencia alcanzada en {k+1} iteraciones (residuo={res:.3e}).")
            return v.reshape((nx, ny)), residuos, k+1

    if verbose:
        print(f"No convergió en {max_iter} iteraciones; residuo final={res:.3e}.")
    return v.reshape((nx, ny)), residuos, max_iter

# Resolver con Gauss–Seidel
v_gs, residuos_gs, iters_gs = gauss_seidel(A, b, max_iter=500, tol=1e-6)

# =============================================================
# 4. SOLUCIÓN DIRECTA (spsolve) PARA COMPARACIÓN
# =============================================================
v_directo = spsolve(A, b).reshape((nx, ny))

diferencia = np.linalg.norm(v_gs - v_directo)
print(f"Diferencia entre Gauss-Seidel y solución directa: {diferencia:.2e}")

# =============================================================
# 5. VISUALIZACIONES BASE
# =============================================================
plt.figure(figsize=(18, 6))

# (a) Convergencia
plt.subplot(1, 3, 1)
plt.semilogy(residuos_gs, marker='o')
plt.xlabel("Iteración")
plt.ylabel("Residuo (log)")
plt.title("Convergencia de Gauss-Seidel")
plt.grid(True, linestyle='--', alpha=0.7)

# (b) Solución Gauss-Seidel
plt.subplot(1, 3, 2)
plt.imshow(v_gs.T, cmap='viridis', origin='lower', aspect='equal')
plt.colorbar(label='v')
plt.title("Solución con Gauss-Seidel")
plt.xlabel("x")
plt.ylabel("y")

# (c) Diferencia GS vs Directo
plt.subplot(1, 3, 3)
plt.imshow(np.abs(v_gs - v_directo).T, cmap='hot', origin='lower', aspect='equal')
plt.colorbar(label='|Δv|')
plt.title("Diferencia: GS vs Directo")
plt.xlabel("x")
plt.ylabel("y")

plt.tight_layout()
plt.show()