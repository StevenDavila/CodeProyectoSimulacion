import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# =============================================
# 1. Configuración del problema
# =============================================
nx, ny = 80, 8
n = nx * ny
left_wall, right_wall, top_wall, bottom_wall = 1.0, 0.0, 0.0, 0.0

# =============================================
# 2. Construcción del sistema no lineal
# =============================================
def build_system(v):
    """Construye la matriz A y el vector b para un estado v dado."""
    A = lil_matrix((n, n))
    b = np.zeros(n)
    v_flat = v.flatten()

    for i in range(nx):
        for j in range(ny):
            idx = i * ny + j

            # Términos no lineales (ejemplo: 0.1*v²)
            non_linear = 0.1 * v_flat[idx]**2

            # Diagonal principal
            A[idx, idx] = -8 + non_linear

            # Vecinos (con condiciones de frontera)
            if i < nx - 1:
                A[idx, (i+1)*ny + j] = 1  # Derecho
            if i > 0:
                A[idx, (i-1)*ny + j] = 3  # Izquierdo
            if j < ny - 1:
                A[idx, i*ny + (j+1)] = 1  # Superior
            if j > 0:
                A[idx, i*ny + (j-1)] = 3  # Inferior

            # Condiciones de frontera
            if i == 0:
                b[idx] = -3 * left_wall - non_linear

    return csr_matrix(A), b

# =============================================
# 3. Método de Newton-Raphson
# =============================================
def newton_raphson(max_iter=50, tol=1e-6):
    v = np.zeros((nx, ny))  # Solución inicial
    residuos = []

    for k in range(max_iter):
        A, b = build_system(v)
        F = A.dot(v.flatten()) - b
        residuo = np.linalg.norm(F)
        residuos.append(residuo)

        print(f"Iteración {k+1}: Residuo = {residuo:.4e}")

        if residuo < tol:
            print(f"¡Convergencia alcanzada en {k+1} iteraciones!")
            break

        delta_v = spsolve(A, -F)
        v += delta_v.reshape((nx, ny))

    return v, residuos

# =============================================
# 4. Ejecución y visualización
# =============================================
v_sol, residuos = newton_raphson(max_iter=20, tol=1e-6)

plt.figure(figsize=(15, 5))

# Gráfico de convergencia
plt.subplot(1, 3, 1)
plt.semilogy(residuos, 'bo-')
plt.xlabel("Iteración")
plt.ylabel("Residuo (log)")
plt.title("Convergencia de Newton-Raphson")
plt.grid()

# Solución final
plt.subplot(1, 3, 2)
plt.imshow(v_sol.T, cmap='viridis', origin='lower', aspect='auto')
plt.colorbar(label='Altura del líquido (v)')
plt.title("Solución Final")

# Evolución del residuo
plt.subplot(1, 3, 3)
plt.plot(residuos, 'r-')
plt.xlabel("Iteración")
plt.ylabel("Residuo")
plt.title("Evolución del Residuo")
plt.grid()

plt.tight_layout()
plt.show()