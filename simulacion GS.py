import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix

# =============================================================
# 1. CONFIGURACIÓN DEL PROBLEMA
# =============================================================
nx, ny = 80, 8
n = nx * ny
Lx, Ly = 1.0, 1.0
x_phys = np.linspace(0.0, Lx, nx)
y_phys = np.linspace(0.0, Ly, ny)

left_wall = 1.0
right_wall = top_wall = bottom_wall = 0.0

# =============================================================
# 2. CONSTRUCCIÓN DEL SISTEMA LINEAL
# =============================================================
def build_system():
    A = lil_matrix((n, n), dtype=float)
    b = np.zeros(n, dtype=float)

    for i in range(nx):
        for j in range(ny):
            idx = i * ny + j
            A[idx, idx] = -8.0

            if i < nx - 1:
                A[idx, (i+1)*ny + j] = 1.0
            if i > 0:
                A[idx, (i-1)*ny + j] = 3.0
            else:
                b[idx] -= 3.0 * left_wall

            if j < ny - 1:
                A[idx, i*ny + (j+1)] = 1.0
            if j > 0:
                A[idx, i*ny + (j-1)] = 3.0
            else:
                b[idx] -= 3.0 * bottom_wall

    return csr_matrix(A), b

A, b = build_system()

# =============================================================
# 3. MÉTODO GAUSS–SEIDEL
# =============================================================
def gauss_seidel(A, b, max_iter=100, tol=1e-4, x0=None, verbose=True):
    n_unknowns = b.size
    v = np.zeros(n_unknowns, dtype=float) if x0 is None else np.asarray(x0, dtype=float).copy()
    diag = A.diagonal()

    for k in range(max_iter):
        for i in range(n_unknowns):
            row_start = A.indptr[i]
            row_end = A.indptr[i+1]
            cols = A.indices[row_start:row_end]
            vals = A.data[row_start:row_end]
            mask = (cols != i)
            sum_ax = np.dot(vals[mask], v[cols[mask]])
            v[i] = (b[i] - sum_ax) / diag[i]

        r = A.dot(v) - b
        res = np.linalg.norm(r)

        if res < tol:
            if verbose:
                print(f"Convergencia alcanzada en {k+1} iteraciones (residuo={res:.3e}).")
            return v.reshape((nx, ny))

    if verbose:
        print(f"No convergió en {max_iter} iteraciones; residuo final={res:.3e}.")
    return v.reshape((nx, ny))

v_gs = gauss_seidel(A, b, max_iter=500, tol=1e-6)

# =============================================================
# 4. VISUALIZACIÓN DE RESULTADO GAUSS–SEIDEL
# =============================================================
plt.figure(figsize=(6, 5))
plt.imshow(v_gs.T, cmap='viridis', origin='lower', aspect='equal',
           extent=[x_phys[0], x_phys[-1], y_phys[0], y_phys[-1]])
plt.colorbar(label='v')
plt.title("Solución con Gauss-Seidel")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()