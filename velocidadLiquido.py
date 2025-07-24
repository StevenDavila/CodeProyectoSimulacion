import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# =============================================
# 1. Configuración del problema (igual que antes)
# =============================================
nx, ny = 80, 8  # Tamaño de la malla (80 en x, 8 en y)
left_wall = 1.0  # Altura en x=0 (h = 1)
right_wall = top_wall = bottom_wall = 0.0  # Otras fronteras (h = 0)

# =============================================
# 2. Construir y resolver el sistema (igual que antes)
# =============================================
def build_system():
    n = nx * ny
    A = lil_matrix((n, n))
    b = np.zeros(n)
    for i in range(nx):
        for j in range(ny):
            idx = i * ny + j
            A[idx, idx] = -8  # Término diagonal
            # Vecinos en x
            if i < nx - 1: A[idx, (i+1)*ny + j] = 1
            if i > 0: A[idx, (i-1)*ny + j] = 3
            elif i == 0: b[idx] -= 3 * left_wall  # Condición x=0 (h=1)
            # Vecinos en y
            if j < ny - 1: A[idx, i*ny + (j+1)] = 1
            if j > 0: A[idx, i*ny + (j-1)] = 3
            elif j == 0: b[idx] -= 3 * bottom_wall
    return csr_matrix(A), b

A, b = build_system()
v_directo = spsolve(A, b).reshape((nx, ny))  # Solución directa

# =============================================
# 3. Calcular altura promedio en cada x (eje 0 a 80)
# =============================================
altura_promedio_x = np.mean(v_directo, axis=1)  # Promedio sobre y
distancias_x = np.arange(nx)  # Eje x de 0 a 79 (índices enteros)

# =============================================
# 4. Graficar Altura vs. Distancia (0 a 80)
# =============================================
plt.figure(figsize=(12, 5))
plt.plot(distancias_x, altura_promedio_x, 'b-', linewidth=2, label='Altura $h(x)$')
plt.scatter(distancias_x, altura_promedio_x, color='red', s=20, label='Puntos de malla')  # Opcional: puntos discretos

# Destacar condiciones de frontera
plt.axvline(x=0, color='k', linestyle='--', linewidth=1, label='$x=0$ ($h=1$)')
plt.axvline(x=nx-1, color='k', linestyle=':', linewidth=1, label='$x=79$ ($h=0$)')
plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

# Ajustes visuales
plt.xlabel("Distancia $x$ (0 a 79)", fontsize=12)
plt.ylabel("velocidad promedio $h(x)$", fontsize=12)
plt.title("velocidad del Líquido a lo largo del Canal (nx=80)", fontsize=14)
plt.xticks(np.arange(0, nx, 5))  # Marcas cada 5 unidades
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()