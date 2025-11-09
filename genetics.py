

import math
import random
import copy
import time
from typing import List, Tuple, Dict

import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False

# ---------------------- Problem definition ----------------------

DEPOT = (20.0, 120.0)
VEHICLE_CAPACITY = 60.0  # kg (initial problem)

ITEM_WEIGHTS = {
    1: 1.2,
    2: 3.8,
    3: 7.5,
    4: 0.9,
    5: 15.4,
    6: 12.1,
    7: 4.3,
    8: 19.7,
    9: 8.6,
    10: 2.5
}

# Provided instance: customers with coordinates and orders (itemID: quantity)
CUSTOMERS_BASE = {
    1: { 'pos': (35.0, 115.0), 'order': {3:2, 1:3} },
    2: { 'pos': (50.0, 140.0), 'order': {2:6} },
    3: { 'pos': (70.0, 100.0), 'order': {7:4, 5:2} },
    4: { 'pos': (40.0, 80.0),  'order': {3:8} },
    5: { 'pos': (25.0, 60.0),  'order': {6:5, 9:2} }
}

# ---------------------- Utility functions ----------------------

def euclidean(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])


def customer_weight(order: Dict[int,int]) -> float:
    return sum(ITEM_WEIGHTS[i]*q for i,q in order.items())

# Precompute weights for base customers
for cid, info in CUSTOMERS_BASE.items():
    info['weight'] = customer_weight(info['order'])

# ---------------------- Scenario generator ----------------------

def generate_random_instance(n_customers:int, n_items:int=10, seed=None) -> Dict:
    """Generate a random instance for testing.
    Coordinates in [0,200] square. Items with random weights based on ITEM_WEIGHTS
    (uses same item set but selects random items per order).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    customers = {}
    for cid in range(1, n_customers+1):
        x = random.uniform(0,200)
        y = random.uniform(0,200)
        n_order_items = random.randint(1, min(4, n_items))
        items = random.sample(list(ITEM_WEIGHTS.keys()), n_order_items)
        order = {it: random.randint(1,6) for it in items}
        customers[cid] = { 'pos': (x,y), 'order': order, 'weight': customer_weight(order) }
    return customers

# ---------------------- Decoding & fitness ----------------------


def decode_chromosome(chromosome: List[int], customers: Dict[int,Dict], vehicle_capacity: float) -> List[List[int]]:
    """
    Decode permutation chromosome into a list of routes (each route is a list of customer ids).
    Greedy split: scan permutation and append to current route until adding next customer would exceed capacity.
    Start new route in that case.
    """
    routes = []
    current_route = []
    current_load = 0.0
    for c in chromosome:
        w = customers[c]['weight']
        if current_route and (current_load + w > vehicle_capacity):
            routes.append(current_route)
            current_route = []
            current_load = 0.0
        current_route.append(c)
        current_load += w
    if current_route:
        routes.append(current_route)
    return routes


def route_distance(route: List[int], customers: Dict[int,Dict]) -> float:
    if not route:
        return 0.0
    dist = 0.0
    pos = DEPOT
    for cid in route:
        dist += euclidean(pos, customers[cid]['pos'])
        pos = customers[cid]['pos']
    dist += euclidean(pos, DEPOT)
    return dist


def total_distance(routes: List[List[int]], customers: Dict[int,Dict]) -> float:
    return sum(route_distance(r, customers) for r in routes)


def capacity_overflow(routes: List[List[int]], customers: Dict[int,Dict], vehicle_capacity: float) -> float:
    overflow = 0.0
    for r in routes:
        load = sum(customers[c]['weight'] for c in r)
        if load > vehicle_capacity:
            overflow += (load - vehicle_capacity)
    return overflow


def fitness(chromosome: List[int], customers: Dict[int,Dict], vehicle_capacity: float,
            penalty_per_kg: float = 1000.0) -> Tuple[float, Dict]:
    """
    Returns fitness value (lower is better) and a dictionary with detailed metrics.
    Fitness = total_distance + penalty_per_kg * overflow_weight
    We keep it as minimization problem. Caller can invert for GA frameworks expecting maximization.
    """
    routes = decode_chromosome(chromosome, customers, vehicle_capacity)
    dist = total_distance(routes, customers)
    overflow = capacity_overflow(routes, customers, vehicle_capacity)
    pen = penalty_per_kg * overflow
    score = dist + pen
    details = { 'routes': routes, 'distance': dist, 'overflow': overflow, 'penalty': pen }
    return score, details

# ---------------------- Genetic operators ----------------------

def tournament_selection(pop: List[List[int]], pop_fitness: List[float], k=3) -> List[int]:
    selected = random.sample(range(len(pop)), k)
    best = min(selected, key=lambda i: pop_fitness[i])
    return copy.deepcopy(pop[best])


def order_crossover(p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    def ox(parent_a, parent_b):
        child = [-1]*n
        child[a:b+1] = parent_a[a:b+1]
        fill = [x for x in parent_b if x not in child]
        idx = 0
        for i in range(n):
            if child[i] == -1:
                child[i] = fill[idx]
                idx += 1
        return child
    return ox(p1,p2), ox(p2,p1)


def swap_mutation(chromosome: List[int], prob: float) -> List[int]:
    ch = chromosome[:]
    for i in range(len(ch)):
        if random.random() < prob:
            j = random.randrange(len(ch))
            ch[i], ch[j] = ch[j], ch[i]
    return ch


def inversion_mutation(chromosome: List[int], prob: float) -> List[int]:
    if random.random() < prob:
        a, b = sorted(random.sample(range(len(chromosome)), 2))
        child = chromosome[:a] + list(reversed(chromosome[a:b+1])) + chromosome[b+1:]
        return child
    return chromosome[:]

# ---------------------- GA main loop ----------------------

class GeneticAlgorithmVRP:
    def __init__(self, customers: Dict[int,Dict], vehicle_capacity: float = VEHICLE_CAPACITY,
                 population_size: int = 200, generations: int = 500,
                 crossover_rate: float = 0.8, mutation_rate: float = 0.2,
                 elitism: int = 1, tournament_k: int = 3, penalty_per_kg: float = 1000.0,
                 seed: int = None):
        self.customers = customers
        self.customer_ids = list(customers.keys())
        self.N = len(self.customer_ids)
        self.vehicle_capacity = vehicle_capacity
        self.pop_size = population_size
        self.generations = generations
        self.cx_rate = crossover_rate
        self.mut_rate = mutation_rate
        self.elitism = elitism
        self.tournament_k = tournament_k
        self.penalty_per_kg = penalty_per_kg
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def initial_population(self) -> List[List[int]]:
        pop = []
        # include a couple heuristics: nearest neighbor starting from depot for different seeds
        for t in range(min(5, self.pop_size)):
            seq = self.nearest_neighbor_permutation(seed=t)
            pop.append(seq)
        # rest random permutations
        while len(pop) < self.pop_size:
            perm = self.customer_ids[:]
            random.shuffle(perm)
            pop.append(perm)
        return pop

    def nearest_neighbor_permutation(self, seed=0) -> List[int]:
        remaining = set(self.customer_ids)
        cur = DEPOT
        order = []
        while remaining:
            best = min(remaining, key=lambda c: euclidean(cur, self.customers[c]['pos']))
            order.append(best)
            remaining.remove(best)
            cur = self.customers[best]['pos']
        return order

    def evaluate_population(self, pop: List[List[int]]) -> List[float]:
        fitnesses = []
        for ind in pop:
            f, _ = fitness(ind, self.customers, self.vehicle_capacity, self.penalty_per_kg)
            fitnesses.append(f)
        return fitnesses

    def evolve(self, verbose: bool = True) -> Dict:
        pop = self.initial_population()
        fitnesses = self.evaluate_population(pop)
        best_history = []
        mean_history = []
        start_time = time.time()

        for gen in range(self.generations):
            new_pop = []
            # --- elitism ---
            elite_idxs = sorted(range(len(pop)), key=lambda i: fitnesses[i])[:self.elitism]
            for i in elite_idxs:
                new_pop.append(copy.deepcopy(pop[i]))

            # --- offspring generation ---
            while len(new_pop) < self.pop_size:
                p1 = tournament_selection(pop, fitnesses, k=self.tournament_k)
                p2 = tournament_selection(pop, fitnesses, k=self.tournament_k)
                if random.random() < self.cx_rate:
                    c1, c2 = order_crossover(p1, p2)
                else:
                    c1, c2 = p1[:], p2[:]

                # --- more aggressive mutation ---
                c1 = swap_mutation(c1, self.mut_rate)
                c1 = inversion_mutation(c1, self.mut_rate)
                c2 = swap_mutation(c2, self.mut_rate)
                c2 = inversion_mutation(c2, self.mut_rate)

                new_pop.append(c1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2)

            pop = new_pop
            fitnesses = self.evaluate_population(pop)

            best_f = min(fitnesses)
            mean_f = np.mean(fitnesses)
            best_idx = min(range(len(pop)), key=lambda i: fitnesses[i])
            best_ind = pop[best_idx]
            best_details = fitness(best_ind, self.customers, self.vehicle_capacity, self.penalty_per_kg)[1]

            best_history.append((gen, best_f, best_details))
            mean_history.append(mean_f)

            if verbose and (gen % max(1, self.generations // 10) == 0):
                print(f"Gen {gen:04d}: best={best_f:.2f}, mean={mean_f:.2f}, dist={best_details['distance']:.2f}, overflow={best_details['overflow']:.2f}, routes={len(best_details['routes'])}")

            # Optional random immigrants every 30 generations to avoid stagnation
            if gen % 30 == 0 and gen > 0:
                n_imm = int(0.05 * self.pop_size)
                for _ in range(n_imm):
                    immigrant = self.customer_ids[:]
                    random.shuffle(immigrant)
                    pop[random.randint(0, self.pop_size - 1)] = immigrant

        end_time = time.time()
        best_gen, best_f, best_details = min(best_history, key=lambda x: x[1])

        result = {
            'best_fitness': best_f,
            'best_routes': best_details['routes'],
            'best_distance': best_details['distance'],
            'best_overflow': best_details['overflow'],
            'time': end_time - start_time,
            'history': best_history,
            'mean_history': mean_history
        }
        return result


# ---------------------- Experiment runner & scenarios ----------------------

def make_base_customers_copy():
    return copy.deepcopy(CUSTOMERS_BASE)


def scenario_list():
    # Several scenarios with increasing difficulty
    scenarios = []
    # 1) Provided dataset: single vehicle, 5 customers (note: some customers may exceed capacity)
    scenarios.append(("base_5_customers_vehicle1", make_base_customers_copy(), VEHICLE_CAPACITY))
    # 2) Add a second identical vehicle by increasing capacity per vehicle (simulate 2 vehicles by allowing split lower capacity?)
    #    For simplicity we model multi-vehicle by using same vehicle_capacity but decoder will create as many routes as needed.
    scenarios.append(("base_5_customers_cap80", make_base_customers_copy(), 80.0))
    # 3) 10 random customers
    scenarios.append(("random_10", generate_random_instance(10, seed=42), 60.0))
    # 4) 25 random customers
    scenarios.append(("random_25", generate_random_instance(25, seed=123), 60.0))
    # 5) Heavier items, smaller capacity (stress test)
    heavy_customers = generate_random_instance(12, seed=7)
    scenarios.append(("random_12_heavy_smallcap", heavy_customers, 40.0))
    return scenarios

# ---------------------- Simple test / demo ----------------------

def demo_run():
    scenarios = scenario_list()
    for name, customers, cap in scenarios:
        print('\n' + '='*60)
        print(f"Scenario: {name} (n_customers={len(customers)}, vehicle_capacity={cap})")
        ga = GeneticAlgorithmVRP(customers, vehicle_capacity=cap,
                                 population_size=200, generations=300,
                                 crossover_rate=0.85, mutation_rate=0.2,
                                 elitism=2, tournament_k=3, penalty_per_kg=1000.0, seed=1)
        res = ga.evolve(verbose=True)
        print(f"Result: best_distance={res['best_distance']:.3f}, overflow={res['best_overflow']:.3f}, routes={len(res['best_routes'])}, time={res['time']:.2f}s")
        for i, r in enumerate(res['best_routes'], 1):
            load = sum(customers[c]['weight'] for c in r)
            print(f"  Route {i}: {r} load={load:.2f} dist={route_distance(r, customers):.2f}")
        # optionally plot progress if matplotlib available
        if _HAS_MATPLOTLIB:
            gens = [g for (g, _, _) in res['history']]
            vals = [f for (_, f, _) in res['history']]
            mean_vals = res.get('mean_history', [])

            plt.figure(figsize=(6, 3))
            plt.plot(gens, vals, label="Best fitness")
            if mean_vals:
                plt.plot(gens, mean_vals, label="Mean fitness", linestyle="--")
            plt.title(f"{name} fitness history")
            plt.xlabel("Generation")
            plt.ylabel("Fitness")
            plt.legend()
            plt.tight_layout()
            fname = f"fitness_{name}.png"
            plt.savefig(fname)
            plt.close()
            print(f"Saved fitness plot: {fname}")


if __name__ == '__main__':
    demo_run()

# End of file
