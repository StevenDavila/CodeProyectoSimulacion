import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import seaborn as sns  # Para gráficos de matrices

# =============================================
# 1. Configuración del problema
# =============================================
nx, ny = 80, 8  # Dominio: x ∈ [0, 79], y ∈ [0, 7]
n = nx * ny      # Número total de nodos

# Condiciones de frontera
left_wall = 1.0  # v(-1, y) = 1
right_wall = 0.0  # v(80, y) = 0
top_wall = 0.0    # v(x, 8) = 0
bottom_wall = 0.0 # v(x, -1) = 0

# =============================================
# 2. Construcción de la matriz A y vector b
# =============================================
A = lil_matrix((n, n))  # Matriz dispersa (eficiente para grandes sistemas)
b = np.zeros(n)  # Crea un vector de ceros de tamaño n

def get_index(i, j):
    """Convierte coordenadas (i,j) a índice lineal."""
    return i * ny + j

for i in range(nx):
    for j in range(ny):
        idx = get_index(i, j)

        # -----------------------------------------
        # Ecuaciones según el tipo de nodo
        # -----------------------------------------
        # Esquinas
        if i == 0 and j == 0:  # Esquina inferior-izquierda (0,0)
            A[idx, get_index(i+1, j)] = 1
            A[idx, idx] = -8
            A[idx, get_index(i, j+1)] = 1
            b[idx] = -3 * left_wall  # v(-1,0) = 1 → 3*1 = 3

        elif i == 0 and j == ny-1:  # Esquina superior-izquierda (0,7)
            A[idx, get_index(i+1, j)] = 1
            A[idx, idx] = -8
            A[idx, get_index(i, j-1)] = 3
            b[idx] = -3 * left_wall  # v(-1,7) = 1 → 3*1 = 3

        elif i == nx-1 and j == 0:  # Esquina inferior-derecha (79,0)
            A[idx, idx] = -8
            A[idx, get_index(i-1, j)] = 3
            A[idx, get_index(i, j+1)] = 1
            # v(80,0) = 0 → se ignora

        elif i == nx-1 and j == ny-1:  # Esquina superior-derecha (79,7)
            A[idx, idx] = -8
            A[idx, get_index(i-1, j)] = 3
            A[idx, get_index(i, j-1)] = 3
            # v(80,7) = 0 → se ignora

        # Lados
        elif i == 0:  # Lado izquierdo (0, y)
            A[idx, get_index(i+1, j)] = 1
            A[idx, idx] = -8
            A[idx, get_index(i, j+1)] = 1
            A[idx, get_index(i, j-1)] = 3
            b[idx] = -3 * left_wall

        elif i == nx-1:  # Lado derecho (79, y)
            A[idx, idx] = -8
            A[idx, get_index(i-1, j)] = 3
            A[idx, get_index(i, j+1)] = 1
            A[idx, get_index(i, j-1)] = 3

        elif j == 0:  # Lado inferior (x, 0)
            A[idx, get_index(i+1, j)] = 1
            A[idx, idx] = -8
            A[idx, get_index(i-1, j)] = 3
            A[idx, get_index(i, j+1)] = 1

        elif j == ny-1:  # Lado superior (x, 7)
            A[idx, get_index(i+1, j)] = 1
            A[idx, idx] = -8
            A[idx, get_index(i-1, j)] = 3
            A[idx, get_index(i, j-1)] = 3

        # Puntos interiores (centro)
        else:
            A[idx, get_index(i+1, j)] = 1
            A[idx, idx] = -8
            A[idx, get_index(i-1, j)] = 3
            A[idx, get_index(i, j+1)] = 1
            A[idx, get_index(i, j-1)] = 3

# =============================================
# 3. Resolución del sistema lineal (Newton-Raphson implícito)
# =============================================
A = csr_matrix(A)  # Convertir a formato eficiente
v = spsolve(A, b).reshape((nx, ny))

# =============================================
# 4. Gráficos
# =============================================
plt.figure(figsize=(15, 5))

# Gráfico 1: Estructura de la matriz A (Jacobiana)
plt.subplot(1, 3, 1)
plt.spy(A, markersize=0.5)
plt.title("Estructura de la Matriz Jacobiana (A)")
plt.xlabel("Nodos")
plt.ylabel("Nodos")

# Gráfico 2: Distribución del líquido
plt.subplot(1, 3, 2)
plt.imshow(v.T, cmap='viridis', origin='lower', aspect='auto')
plt.colorbar(label='Altura del líquido (v)')
plt.title("Distribución del Líquido")
plt.xlabel("Coordenada x")
plt.ylabel("Coordenada y")

# Gráfico 3: Corte transversal (ejemplo en y=4)
plt.subplot(1, 3, 3)
y_cut = 4
plt.plot(v[:, y_cut], 'r-', linewidth=2)
plt.title(f"Corte Transversal en y={y_cut}")
plt.xlabel("Coordenada x")
plt.ylabel("v(x, y)")
plt.grid()

plt.tight_layout()
plt.show()