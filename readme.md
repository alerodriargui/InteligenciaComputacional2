# Vehicle Routing Problem (VRP) — Algoritmo Genético

## Resumen

Proyecto de ejemplo que implementa un algoritmo genético simple para optimizar rutas de entrega (VRP) para un único depósito y una flota de vehículos con capacidad limitada. Este script está pensado como punto de partida para la práctica `Assignment 2: Delivery route optimization using genetic algorithms` (Inteligencia Computacional — curso 25/26).



## Datos iniciales (ejemplo)
- Depósito: `[20, 120]`
- Clientes (coordenadas):
  - `[35, 115]`, `[50, 140]`, `[70, 100]`, `[40, 80]`, `[25, 60]`
- Pesos de los ítems: `[1.2, 3.8, 7.5, 0.9, 15.4, 12.1, 4.3, 19.7, 8.6, 2.5]`
- Pedidos por cliente (ítem, cantidad):
  - Cliente 1: `[(3,2),(1,3)]` → demanda calculada en kg
  - Cliente 2: `[(2,6)]`
  - Cliente 3: `[(7,4),(5,2)]`
  - Cliente 4: `[(3,8)]`
  - Cliente 5: `[(6,5),(9,2)]`
- Capacidad del vehículo: `60 kg`

> El script calcula automáticamente la demanda total por cliente y separa clientes "válidos" (demandas ≤ capacidad) e "inválidos" (demandas > capacidad). Los clientes inválidos no se sirven en esta versión simple.


## Requisitos
- Python 3.8+
- Paquetes:
  - `matplotlib`

Instalación rápida (virtualenv recomendado):

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
pip install matplotlib
```


Salida esperada:
- Número de clientes válidos e inválidos
- Lista de demandas válidas/invalidas
- Mejor orden de visita (permutación de índices de clientes válidos)
- Distancia total de la solución
- Listado de viajes resultantes y carga por viaje
- Gráfica con las rutas y la carga por viaje


## Estructura del script
- **Cálculo de demandas:** lee `pedidos` y multiplica por `pesos` para obtener `demandas` por cliente.
- **Filtrado por capacidad:** separa clientes válidos e inválidos.
- **Funciones auxiliares:**
  - `distancia(a, b)` — distancia euclídea entre puntos.
  - `dividir_en_rutas(ruta)` — dada una permutación de índices construye los viajes respetando la capacidad.
  - `distancia_total(ruta)` — suma distancias de todos los viajes (vuelta al depósito incluida).
  - `fitness(ruta)` — función objetivo (minimizar distancia total).
- **Operadores genéticos:**
  - `crear_poblacion(n, n_clientes)` — población inicial con permutaciones aleatorias.
  - `seleccion(poblacion, fitnesses)` — torneo de tamaño 2.
  - `cruce(p1, p2)` — operador de ordenación (crossover por sección + rellenado en orden del segundo progenitor).
  - `mutacion(ruta, prob)` — swap aleatorio con probabilidad `prob`.
- **Algoritmo principal:** `algoritmo_genetico(n_generaciones, tam_pob)` que devuelve la mejor solución encontrada.



## Primer experimento (Problema inicial)

- Tamaño de población: `20, 50, 100`
- Número de generaciones: `100, 500, 2000`
- Probabilidad de mutación: `0.01, 0.05, 0.1, 0.2`
- Métodos de selección: torneo (k=2,3), ruleta, ranking
- Cruces alternativos: OX (Order Crossover)


## Uso
Ejecuta el script:

```bash
python p2Basico.py
```

## Resultados
```
Clientes válidos: 4
Demandas válidas: [18.6, 22.8, 48.0, 60.0]
Orden de visita: [2, 3, 0, 1]
Distancia total: 278.17
Clientes inválidos: 1
Demandas inválidas: [77.7]
Viaje 1: [2]  (Carga 48.0 kg)
Viaje 2: [3]  (Carga 60.0 kg)
Viaje 3: [0, 1]  (Carga 41.4 kg)
```

## Observaciones
Estos resultados muestran cómo el algoritmo filtra primero a los clientes cuya demanda excede la capacidad máxima del vehículo. En este caso, 4 clientes son válidos y 1 cliente queda descartado por superar la capacidad permitida (77.7 kg).
Después, el sistema genera un orden de visita eficiente para los clientes admitidos y calcula la distancia total recorrida (278.17 unidades).

El reparto final se divide en 3 viajes independientes, porque la suma de las demandas, respetando el orden propuesto, supera la capacidad si se intentara agruparlos en un único recorrido. Por eso:

Viaje 1: solo el cliente 2 (48.0 kg)

Viaje 2: solo el cliente 3 (60.0 kg, muy próximo al límite)

Viaje 3: clientes 0 y 1 (41.4 kg combinados), que sí pueden ir juntos


## Segundo experimento (Capacidad = 80 kg)
En este experimento modificamos la capacidad del vehículo a 80 kg, ya que un cliente quedaba descartado en el experimento original. 

- Tamaño de población: `20, 50, 100`
- Número de generaciones: `100, 500, 2000`
- Probabilidad de mutación: `0.01, 0.05, 0.1, 0.2`
- Métodos de selección: torneo (k=2,3), ruleta, ranking
- Cruces alternativos: OX (Order Crossover)


## Uso
Ejecuta el script:

```bash
python p2Capacity80.py
```

## Observaciones
En este escenario todos los clientes son considerados válidos, incluso aquellos cuya demanda supera el límite habitual de capacidad. Esto permite trabajar con un caso “sin restricciones”, útil para comparar el efecto que tiene imponer un límite de carga por vehículo.

El algoritmo genera un orden de visita que cubre a los 5 clientes y calcula una distancia total de 350.93 unidades, algo mayor que en el ejemplo restringido debido a que hay más paradas obligatorias.

Al organizar los viajes, se observa que las cargas resultantes pueden superar los valores que normalmente serían aceptables en un vehículo real (como en el tercer viaje con 78.6 kg). Esto ocurre porque en este ejemplo no se aplica ningún filtro por capacidad: el objetivo es únicamente construir rutas válidas en términos de recorrido.

Los viajes quedan así:

Viaje 1: clientes 2 y 1 (70.8 kg)

Viaje 2: cliente 4 (77.7 kg)

Viaje 3: clientes 3 y 0 (78.6 kg)




