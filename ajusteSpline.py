import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import CubicSpline

# =============================================
# 1. CONFIGURACIÓN DEL PROBLEMA
# =============================================
nx, ny = 80, 8   # Malla 80 x 8
n = nx * ny

left_wall  = 1.0
right_wall = 0.0
top_wall   = 0.0
bottom_wall= 0.0

# =============================================
# 2. CONSTRUCCIÓN DEL SISTEMA
# =============================================
def build_system():
    A = lil_matrix((n, n))
    b = np.zeros(n)

    for i in range(nx):
        for j in range(ny):
            idx = i * ny + j
            A[idx, idx] = -8.0

            # Vecinos en x
            if i < nx - 1:
                A[idx, (i + 1) * ny + j] = 1.0
            else:
                b[idx] -= 1.0 * right_wall

            if i > 0:
                A[idx, (i - 1) * ny + j] = 3.0
            else:
                b[idx] -= 3.0 * left_wall

            # Vecinos en y
            if j < ny - 1:
                A[idx, i * ny + (j + 1)] = 1.0
            else:
                b[idx] -= 1.0 * top_wall

            if j > 0:
                A[idx, i * ny + (j - 1)] = 3.0
            else:
                b[idx] -= 3.0 * bottom_wall
    return csr_matrix(A), b

A, b = build_system()

# =============================================
# 3. MÉTODO GAUSS-SEIDEL
# =============================================
def gauss_seidel(A, b, max_iter=500, tol=1e-6, omega=1.0):
    n = len(b)
    v = np.zeros(n)
    residuos = []

    for k in range(max_iter):
        for i in range(n):
            start, end = A.indptr[i], A.indptr[i + 1]
            row_idx, row_val = A.indices[start:end], A.data[start:end]
            diag_mask = (row_idx == i)
            aii = row_val[diag_mask][0]
            sum_ax = np.dot(row_val[~diag_mask], v[row_idx[~diag_mask]])
            v_new_i = (b[i] - sum_ax) / aii
            v[i] = (1 - omega) * v[i] + omega * v_new_i

        residuo = np.linalg.norm(A.dot(v) - b)
        residuos.append(residuo)
        if residuo < tol:
            print(f"Convergencia en {k+1} iteraciones.")
            break
    return v.reshape((nx, ny)), residuos

v_gs, residuos_gs = gauss_seidel(A, b, max_iter=500, tol=1e-6, omega=1.0)

# =============================================
# 4. MÉTODO DIRECTO (COMPARACIÓN)
# =============================================
v_directo = spsolve(A, b).reshape((nx, ny))
print("Diferencia con solución directa:", np.linalg.norm(v_gs - v_directo))

# =============================================
# 5. VISUALIZACIÓN DEL CAMPO ORIGINAL (RECTANGULAR)
# =============================================
plt.figure(figsize=(12, 3), dpi=120)  # Ajuste rectangular
plt.imshow(v_gs.T, origin='lower', cmap='viridis',
           extent=[0, nx-1, 0, ny-1], aspect='auto')
plt.colorbar(label='Valor de v')
plt.title("Distribución del líquido en el dominio")
plt.xlabel("Coordenada x")
plt.ylabel("Coordenada y")
plt.tight_layout()
plt.show()

# =============================================
# 6. SUAVIZADO CON SPLINE CÚBICO (en x)
# =============================================
def suavizar_spline(v, factor=5):
    x_idx = np.arange(nx)
    x_ref = np.linspace(0, nx-1, nx * factor)
    v_suave = np.zeros((x_ref.size, ny))

    for j in range(ny):
        cs = CubicSpline(x_idx, v[:, j], bc_type='natural')
        v_suave[:, j] = cs(x_ref)

    return x_ref, v_suave

x_ref, v_suave = suavizar_spline(v_gs, factor=5)

# =============================================
# 7. COMPARACIÓN: ORIGINAL VS. SUAVIZADO (RECTANGULAR)
# =============================================
plt.figure(figsize=(14, 3), dpi=120)  # Figura horizontal y rectangular

# Campo original
plt.subplot(1, 2, 1)
plt.imshow(v_gs.T, origin='lower', cmap='viridis',
           extent=[0, nx-1, 0, ny-1], aspect='auto')
plt.colorbar(label='v')
plt.title("Campo original (Gauss-Seidel)")
plt.xlabel("Coordenada x")
plt.ylabel("Coordenada y")

# Campo suavizado
plt.subplot(1, 2, 2)
plt.imshow(v_suave.T, origin='lower', cmap='viridis',
           extent=[0, nx-1, 0, ny-1], aspect='auto')
plt.colorbar(label='v')
plt.title("Campo suavizado (Spline cúbico)")
plt.xlabel("Coordenada x (refinada)")
plt.ylabel("Coordenada y")

plt.tight_layout()
plt.show()