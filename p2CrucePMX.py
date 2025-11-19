import random
import math
import matplotlib.pyplot as plt

# --- Datos del problema ----------------------------------------------------

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

# Calcular demanda total (kg)
demandas = []
for pedido in pedidos:
    total = 0
    for item, cant in pedido:
        total += pesos[item - 1] * cant
    demandas.append(total)

capacidad = 60

# Filtrar los clientes que caben
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

# --- Funciones auxiliares ----------------------------------------------------

def distancia(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def dividir_en_rutas(ruta):
    rutas = []
    carga = 0
    actual = []
    for c in ruta:
        if carga + demandas_validas[c] <= capacidad:
            actual.append(c)
            carga += demandas_validas[c]
        else:
            rutas.append(actual)
            actual = [c]
            carga = demandas_validas[c]
    if actual:
        rutas.append(actual)
    return rutas

def distancia_total(ruta):
    rutas = dividir_en_rutas(ruta)
    total = 0
    for r in rutas:
        pos = depot
        for c in r:
            total += distancia(pos, clientes_validos[c])
            pos = clientes_validos[c]
        total += distancia(pos, depot)
    return total

def fitness(ruta):
    return distancia_total(ruta)

# --- Operadores Genéticos ----------------------------------------------------

# Selección torneo
def seleccion(poblacion, fitnesses):
    i1, i2 = random.sample(range(len(poblacion)), 2)
    return poblacion[i1][:] if fitnesses[i1] < fitnesses[i2] else poblacion[i2][:]

# ---------- 1. OX (Order Crossover) ----------
def cruce_OX(p1, p2):
    a, b = sorted(random.sample(range(len(p1)), 2))
    hijo = [-1] * len(p1)

    hijo[a:b] = p1[a:b]
    pos = b

    for x in p2:
        if x not in hijo:
            if pos == len(p1):
                pos = 0
            hijo[pos] = x
            pos += 1

    return hijo

# ---------- 2. PMX (Partially Mapped Crossover) ----------
def cruce_PMX(p1, p2):
    a, b = sorted(random.sample(range(len(p1)), 2))
    hijo = p1[:]

    mapa = {}

    for i in range(a, b):
        hijo[i] = p2[i]
        mapa[p2[i]] = p1[i]

    for i in range(len(hijo)):
        if i >= a and i < b:
            continue
        while hijo[i] in mapa:
            hijo[i] = mapa[hijo[i]]

    return hijo

# ---------- 3. CX (Cycle Crossover) ----------
def cruce_CX(p1, p2):
    hijo = [-1] * len(p1)

    index = 0
    ciclo = set()

    while index not in ciclo:
        ciclo.add(index)
        index = p1.index(p2[index])

    for i in range(len(p1)):
        hijo[i] = p1[i] if i in ciclo else p2[i]

    return hijo

# ---------- Mutación ----------
def mutacion(ruta, prob=0.1):
    if random.random() < prob:
        i, j = random.sample(range(len(ruta)), 2)
        ruta[i], ruta[j] = ruta[j], ruta[i]

# --- Algoritmo Genético -----------------------------------------------------

def algoritmo_genetico(n_generaciones=200, tam_pob=50, tipo_cruce="OX"):

    # Selección de operador
    operadores = {
        "OX": cruce_OX,
        "PMX": cruce_PMX,
        "CX": cruce_CX
    }

    cruce_seleccionado = operadores[tipo_cruce]

    poblacion = []
    base = list(range(len(clientes_validos)))
    for _ in range(tam_pob):
        r = base[:]
        random.shuffle(r)
        poblacion.append(r)

    fitnesses = [fitness(r) for r in poblacion]

    for _ in range(n_generaciones):
        nueva = []
        for _ in range(tam_pob):
            p1 = seleccion(poblacion, fitnesses)
            p2 = seleccion(poblacion, fitnesses)

            hijo = cruce_seleccionado(p1, p2)
            mutacion(hijo, 0.1)

            nueva.append(hijo)

        poblacion = nueva
        fitnesses = [fitness(r) for r in poblacion]

    mejor = min(range(tam_pob), key=lambda i: fitnesses[i])
    return poblacion[mejor], fitnesses[mejor]

# --- Ejecución ---------------------------------------------------------------

mejor_ruta, mejor_valor = algoritmo_genetico(tipo_cruce="PMX")  # <-- CAMBIAR OX / PMX / CX

print("Clientes válidos:", len(clientes_validos))
print("Demandas válidas:", [round(x, 1) for x in demandas_validas])
print("Orden de visita:", mejor_ruta)
print("Distancia total:", round(mejor_valor, 2))
print("Clientes inválidos:", len(clientes_invalidos))
print("Demandas inválidas:", [round(x, 1) for x in demandas_invalidas])

# Mostrar viajes
rutas = dividir_en_rutas(mejor_ruta)
for i, r in enumerate(rutas):
    carga = sum(demandas_validas[c] for c in r)
    print(f"Viaje {i+1}: {r}  (Carga {round(carga,1)} kg)")

# --- Plot gráfico ------------------------------------------------------------

def plot_rutas(rutas):
    plt.figure(figsize=(7, 7))

    for i, (x, y) in enumerate(clientes_validos):
        plt.scatter(x, y, c='blue')
        plt.text(x+1, y+1, f'C{i+1}', fontsize=9)

    plt.scatter(depot[0], depot[1], c='red', s=100, label='Depósito')

    colores = ['green', 'orange', 'purple', 'cyan', 'brown']

    for i, r in enumerate(rutas):
        puntos = [depot] + [clientes_validos[c] for c in r] + [depot]
        xs = [p[0] for p in puntos]
        ys = [p[1] for p in puntos]
        plt.plot(xs, ys, '-o', color=colores[i % len(colores)], label=f'Viaje {i+1}')

        carga = sum(demandas_validas[c] for c in r)
        x_text = sum(xs)/len(xs)
        y_text = sum(ys)/len(ys)
        plt.text(x_text, y_text, f"{round(carga,1)} kg", fontsize=10, color=colores[i % len(colores)], fontweight='bold')

    plt.title("Rutas del vehículo (con carga por viaje)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_rutas(rutas)
