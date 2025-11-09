# Assignment 2: Delivery route optimization using genetic algorithms

**11753 - Computational Intelligence. Master in Intelligent Systems. Academic Year 25/26**

---

## File
`genetics.py`

Runnable Python implementation (single-file) of a **Genetic Algorithm (GA)** for the **Vehicle Routing Problem (VRP)** described in the assignment prompt.

---

## How to Run
```bash
python genetics.py
```

---

## Contents
The script contains:
- **Problem definition** and several **test scenarios** (including the provided dataset).
- **GA implementation** (representation, population operators, decoding strategy).
- **Experiments runner** that sweeps parameters and prints/plots results.
- **Simple CLI** at the bottom that runs a default experiment.

---

## Dependencies
Only Python standard library + `numpy` + `matplotlib` (optional for plots).

Install via:
```bash
pip install numpy matplotlib
```

---

## Notes
- **Chromosome:** a permutation of customer IDs. Decoding uses a greedy *split* that creates routes in order by scanning the permutation and starting a new route when the next customer would overflow the vehicle capacity.
- **Penalization:** any capacity violation after decoding is penalised in the fitness by a large penalty times the overflow weight.
- **Operators implemented:**
  - Tournament selection
  - Order Crossover (OX)
  - Swap & inversion mutation
  - Elitism is supported

---