import random
import math
import matplotlib.pyplot as plt

# ---------------- Datos (escenario base) ----------------
random.seed(42)  # reproducibilidad

depot = [20, 120]
clientes = [
    [35, 115],
    [50, 140],
    [70, 100],
    [40, 80],
    [25, 60]
]

pesos = [1.2, 3.8, 7.5, 0.9, 15.4, 12.1, 4.3, 19.7, 8.6, 2.5]

pedidos = [
    [(3, 2), (1, 3)],
    [(2, 6)],
    [(7, 4), (5, 2)],
    [(3, 8)],
    [(6, 5), (9, 2)]
]

# capacidad total por vehículo y número de vehículos
capacidad = 60
num_vehiculos = 2  # ejemplo: puedes variar a 1,2,3...

# ---------------- calcular demandas por cliente ----------------
demandas = []
for pedido in pedidos:
    total = 0
    for item, cant in pedido:
        total += pesos[item - 1] * cant
    demandas.append(total)

# separar clientes que exceden la capacidad individual (hard infeasible)
clientes_validos = []
demandas_validas = []
clientes_invalidos = []
demandas_invalidas = []
for i in range(len(clientes)):
    if demandas[i] <= capacidad:
        clientes_validos.append(clientes[i])
        demandas_validas.append(demandas[i])
    else:
        clientes_invalidos.append(clientes[i])
        demandas_invalidas.append(demandas[i])

n_clientes = len(clientes_validos)

# ---------------- utilidades ----------------
def distancia(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def dividir_en_rutas_por_capacidad_y_vehiculos(ruta_perm):
    """
    Divide la permutación de clientes en rutas respetando capacidad.
    Además, limita al número de vehículos disponibles -> si hace falta más rutas
    devolvemos rutas y un indicador de penalización por exceso de vehículos.
    """
    rutas = []
    carga = 0
    actual = []
    for idx in ruta_perm:
        demanda = demandas_validas[idx]
        if carga + demanda <= capacidad:
            actual.append(idx)
            carga += demanda
        else:
            rutas.append(actual)
            actual = [idx]
            carga = demanda
    if actual:
        rutas.append(actual)

    exceso_vehiculos = max(0, len(rutas) - num_vehiculos)
    return rutas, exceso_vehiculos

def distancia_total(ruta_perm):
    """Suma distancias de todos los viajes (vuelven al depósito)."""
    rutas, _ = dividir_en_rutas_por_capacidad_y_vehiculos(ruta_perm)
    total = 0.0
    for r in rutas:
        pos = depot
        for c in r:
            total += distancia(pos, clientes_validos[c])
            pos = clientes_validos[c]
        total += distancia(pos, depot)
    return total

# ---------------- fitness con penalizaciones ----------------
PENALIZACION_EXCESO_VEHICULOS = 1000.0  # grande para penalizar violación
PENALIZACION_CLIENTE_INDIVIDUAL_INFEASIBLE = 1e6  # si algún cliente individual > capacidad

# si existen clientes que por sí solos exceden la capacidad, forzamos penalización alta
def fitness(ruta_perm):
    # chequeo hard infeasible (cliente individual)
    if clientes_invalidos:
        return PENALIZACION_CLIENTE_INDIVIDUAL_INFEASIBLE + sum(demandas_invalidas)

    dist = distancia_total(ruta_perm)
    _, exceso = dividir_en_rutas_por_capacidad_y_vehiculos(ruta_perm)
    # penalización proporcional al número de vehículos extra y a la carga media por viaje si quieres
    return dist + exceso * PENALIZACION_EXCESO_VEHICULOS

# ---------------- operadores ----------------
def crear_poblacion(n_pob, n_clientes):
    base = list(range(n_clientes))
    poblacion = []
    for _ in range(n_pob):
        r = base[:]
        random.shuffle(r)
        poblacion.append(r)
    return poblacion

def seleccion_torneo(poblacion, fitnesses, k=3):
    aspirantes = random.sample(range(len(poblacion)), k)
    mejor = min(aspirantes, key=lambda i: fitnesses[i])
    return poblacion[mejor][:]

def ox_crossover(p1, p2):
    """Order Crossover (OX) para permutaciones."""
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    hijo = [-1]*n
    # copiar segmento central de p1
    hijo[a:b+1] = p1[a:b+1]
    # rellenar con el orden de p2
    pos = (b+1) % n
    for gene in p2[(b+1):] + p2[:(b+1)]:
        if gene not in hijo:
            hijo[pos] = gene
            pos = (pos+1) % n
    return hijo

def mutacion_swap(ruta, prob_swap=0.2, prob_inversion=0.05):
    # swap mutación con probabilidad
    if random.random() < prob_swap:
        i, j = random.sample(range(len(ruta)), 2)
        ruta[i], ruta[j] = ruta[j], ruta[i]
    # pequeña inversión de segmento
    if random.random() < prob_inversion:
        a, b = sorted(random.sample(range(len(ruta)), 2))
        ruta[a:b+1] = reversed(ruta[a:b+1])

# ---------------- algoritmo genético (mejorado) ----------------
def algoritmo_genetico(n_generaciones=300, tam_pob=80, torneo_k=3,
                       p_elite=0.05, prob_mut_swap=0.2, prob_mut_inv=0.05):
    poblacion = crear_poblacion(tam_pob, n_clientes)
    fitnesses = [fitness(ind) for ind in poblacion]

    n_elite = max(1, int(p_elite * tam_pob))
    mejor_hist = []
    media_hist = []

    for gen in range(n_generaciones):
        nueva = []
        # elitismo: copiar mejores
        indices_orden = sorted(range(len(poblacion)), key=lambda i: fitnesses[i])
        for i in indices_orden[:n_elite]:
            nueva.append(poblacion[i][:])

        # crear hijos hasta cubrir la población
        while len(nueva) < tam_pob:
            p1 = seleccion_torneo(poblacion, fitnesses, k=torneo_k)
            p2 = seleccion_torneo(poblacion, fitnesses, k=torneo_k)
            hijo = ox_crossover(p1, p2)
            mutacion_swap(hijo, prob_swap=prob_mut_swap, prob_inversion=prob_mut_inv)
            nueva.append(hijo)

        poblacion = nueva
        fitnesses = [fitness(ind) for ind in poblacion]
        mejor = min(fitnesses)
        media = sum(fitnesses)/len(fitnesses)
        mejor_hist.append(mejor)
        media_hist.append(media)

        # (opcional) print cada X generaciones
        if gen % 50 == 0 or gen == n_generaciones-1:
            print(f"Gen {gen}: mejor {mejor:.2f} - media {media:.2f}")

    # devolver mejor individuo y su historial para graficar
    mejor_idx = min(range(len(poblacion)), key=lambda i: fitnesses[i])
    return poblacion[mejor_idx], fitnesses[mejor_idx], mejor_hist, media_hist

# ---------------- ejecutar ----------------
mejor_ruta, mejor_valor, hist_mejor, hist_media = algoritmo_genetico(
    n_generaciones=400, tam_pob=120, torneo_k=4, p_elite=0.05,
    prob_mut_swap=0.3, prob_mut_inv=0.08
)

print("\n--- RESULTADO ---")
print("Clientes válidos:", n_clientes)
print("Demandas válidas:", [round(x, 1) for x in demandas_validas])
print("Orden de visita (indices en clientes_validos):", mejor_ruta)
print("Distancia total (fitness):", round(mejor_valor, 2))
print("Clientes inválidos:", len(clientes_invalidos))
if clientes_invalidos:
    print("Demandas inválidas:", [round(x, 1) for x in demandas_invalidas])

# mostrar viajes reales
rutas, exceso = dividir_en_rutas_por_capacidad_y_vehiculos(mejor_ruta)
for i, r in enumerate(rutas):
    carga = sum(demandas_validas[c] for c in r)
    print(f"Viaje {i+1}: {r}  (Carga {round(carga,1)} kg)")

if exceso > 0:
    print(f"AVISO: se requieren {exceso} vehículos adicionales (penalizados).")

# ---------------- gráficos ----------------
def plot_rutas(rutas):
    plt.figure(figsize=(7, 7))
    for i, (x, y) in enumerate(clientes_validos):
        plt.scatter(x, y)
        plt.text(x+1, y+1, f'C{i+1}', fontsize=9)
    plt.scatter(depot[0], depot[1], s=100, marker='s')
    colores = ['green', 'orange', 'purple', 'cyan', 'brown', 'magenta', 'yellow']
    for i, r in enumerate(rutas):
        puntos = [depot] + [clientes_validos[c] for c in r] + [depot]
        xs = [p[0] for p in puntos]
        ys = [p[1] for p in puntos]
        plt.plot(xs, ys, '-o', label=f'Viaje {i+1}', linewidth=1.5)
        carga = sum(demandas_validas[c] for c in r)
        x_text = sum(xs)/len(xs)
        y_text = sum(ys)/len(ys)
        plt.text(x_text, y_text, f"{round(carga,1)} kg", fontsize=9, fontweight='bold')
    # clientes inválidos (si existen)
    for (x, y) in clientes_invalidos:
        plt.scatter(x, y, marker='x')
        plt.text(x+1, y+1, 'Inválido', color='red')
    plt.title("Rutas (visualización)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_rutas(rutas)

