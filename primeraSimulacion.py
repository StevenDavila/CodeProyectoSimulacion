import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve

# Dimensiones
nx, ny = 80, 8
n = nx * ny

# Construcción de la matriz del sistema (A) y vector de términos independientes (b)
A = np.zeros((n, n))
b = np.zeros(n)

def idx(i, j):
    """Convierte coordenadas (i,j) en índice lineal"""
    return i * ny + j

# Construimos la matriz A
for i in range(nx):
    for j in range(ny):
        current_idx = idx(i, j)

        # Condiciones de frontera implícitas
        # Derecha (x=80): v=0 → términos con x+1 se eliminan cuando i=79
        # Izquierda (x=-1): v=1 → términos con x-1 se reemplazan por 3*1 cuando i=0
        # Arriba (y=8): v=0 → términos con y+1 se eliminan cuando j=7
        # Abajo (y=-1): v=0 → términos con y-1 se eliminan cuando j=0

        # Coeficientes
        A[current_idx, current_idx] = -8  # Término central

        # Términos en x
        if i < nx - 1:
            A[current_idx, idx(i+1, j)] = 1
        if i > 0:
            A[current_idx, idx(i-1, j)] = 3
        elif i == 0:
            b[current_idx] -= 3  # Porque v(-1,j) = 1 → 3*1 = 3

        # Términos en y
        if j < ny - 1:
            A[current_idx, idx(i, j+1)] = 1
        if j > 0:
            A[current_idx, idx(i, j-1)] = 3

# Convertimos a formato disperso para eficiencia
A_sparse = csr_matrix(A)

# Resolvemos el sistema lineal Av = b
v = spsolve(A_sparse, b)

# Reorganizamos la solución en forma de matriz 2D
v_matrix = v.reshape((nx, ny))

# Visualización
plt.figure(figsize=(12, 6))
plt.imshow(v_matrix.T, origin='lower', cmap='viridis', aspect='auto')
plt.colorbar(label='Valor de v')
plt.xlabel('Coordenada x')
plt.ylabel('Coordenada y')
plt.title('Distribución del líquido en el dominio')
plt.show()