# 🚚 VRP Experiments with Genetic Algorithm

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-green?style=for-the-badge)

This repository hosts a collection of computational experiments designed to solve variants of the **Vehicle Routing Problem (VRP)** applying **Genetic Algorithms (GA)**.

The project explores how capacity constraints and fleet size impact logistics efficiency, seeking to minimize the total distance traveled to satisfy the demand of a customer portfolio.

---

## 👥 Authors

Work developed for the **Computational Intelligence** course by:

*   **Alejandro Rodríguez Arguimbau**
*   **Soufyane Youbi**
*   **Pau Girón Rodríguez**

---

## 🧬 Technical Algorithm Details

All experiments share a base implementation of a Genetic Algorithm configured as follows:

| Component | Implementation / Value | Description |
| :--- | :--- | :--- |
| **Representation** | Integer Permutation | Ordered list of customer indices (chromosome). |
| **Fitness Function** | Total Euclidean Distance | Sum of distances of all generated routes. |
| **Selection** | Binary Tournament | Two individuals are chosen at random, the one with the shortest distance wins. |
| **Crossover** | Order Crossover (OX) | Maintains relative order to avoid duplicate customers. |
| **Mutation** | Swap | Probability of **10%** ($P_m = 0.1$). |
| **Population** | 50 Individuals | Constant size in each generation. |
| **Generations** | 200 Iterations | Fixed stopping criterion. |
| **Decoding** | *First-Fit* Heuristic | The vehicle is filled to maximum capacity before returning to the depot. |

---

## 📂 Structure and Scenarios

The study is divided into three progressive notebooks. It is recommended to follow this order:

### 1. 🚛 Basic Scenario (`VRP_basico.ipynb`)
*   **Constraint:** 1 Vehicle | Capacity **60 kg**.
*   **Challenge:** Capacity is highly restrictive.
*   **Finding:** Customer 5 (77.7 kg) is **impossible to serve** (demand > capacity). The system marks it as "Invalid".

### 2. 🚛 Increased Capacity Scenario (`VRP_80Capacity.ipynb`)
*   **Constraint:** 1 Vehicle | Capacity **80 kg**.
*   **Improvement:** Capacity is increased to include Customer 5.
*   **Finding:** **100% coverage** is achieved. Load efficiency is maximal (97-99% occupancy), demonstrating that a small capacity increase can unlock major logistical improvements.

### 3. 🚛🚛 Multi-Vehicle Scenario (`2Vehicles_80capacity.ipynb`)
*   **Constraint:** **2 Vehicles** | Capacity **80 kg** each.
*   **Realism:** Simulates a real fleet with multiple shifts.
*   **Finding:** The algorithm distributes necessary trips between the two trucks. A total of 3 trips are required, resulting in an asymmetric operation (one truck does 2 shifts, the other 1).

---

## 📊 Results Comparison Table

| Metric | Capacity 60 kg | Capacity 80 kg | 2-Vehicle Fleet (80 kg) |
| :--- | :--- | :--- | :--- |
| **Customers Served** | 4 / 5 (80%) | **5 / 5 (100%)** | **5 / 5 (100%)** |
| **Demand Served** | 149.4 kg | 227.1 kg | 227.1 kg |
| **Total Trips** | 3 | 3 | 3 (distributed) |
| **Load Efficiency** | Medium (fragmented) | **Very High (saturated)** | Very High |
| **Logistical Solution** | **Infeasible** (Lost customer) | **Optimal** (1 truck) | **Flexible** (2 trucks) |

---

## 🚀 Installation and Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/alerodriargui/InteligenciaComputacional2.git
    cd InteligenciaComputacional2
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the notebooks:**
    Open Jupyter Notebook or VS Code and execute the cells in order.
    ```bash
    jupyter notebook
    ```
