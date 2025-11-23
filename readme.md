# Quick Guide: VRP Experiments with a Genetic Algorithm

This repository contains three notebooks solving a **Vehicle Routing Problem (VRP)** using Genetic Algorithms.

## Authors

This project was developed by the following members of the group:

- Alejandro Rodríguez Arguimbau

- Soufyane Youbi

- Pau Girón Rodríguez

## Notebooks and Execution Order

1. **`VRP_basico.ipynb`** – Single vehicle with 60 kg capacity (default problem).
2. **`VRP_80Capacity.ipynb`** – Single vehicle with 80 kg capacity.
3. **`2Vehicles_80capacity.ipynb`** – Two vehicles with 80 kg capacity.

> Run the notebooks in this order to keep the workflow consistent and enable proper comparison of results.

## Requirements

Install the required libraries:
```bash
pip install -r requirements.txt
```


## Scenario Comparison Table

| Metric / Scenario | Capacity 60 kg | Capacity 80 kg | Multi-Trip (80 kg, 2 Vehicles) |
|------------------|----------------|----------------|---------------------------------|
| **Feasibility** | 4/5 customers served | 5/5 customers served | 5/5 customers served |
| **Invalid Customers** | C5 (77.7 kg) | None | None |
| **Total Demand Served** | 149.4 kg | 227.1 kg | 227.1 kg |
| **Number of Trips** | 3 trips | 3 trips | 3 trips total (V1×2, V2×1) |
| **Vehicle Occupancy** | 60–100% (high fragmentation) | 97–99% (high saturation) | Similar to 80 kg scenario |
| **Route Efficiency** | Low (forced extra routes) | High (consolidated loads) | High, but requires waiting for returns |
| **Fleet Required** | 1 vehicle per trip | 1 vehicle per trip | 2 vehicles, reused in shifts |
| **Operational Cost** | High (many partial loads) | Lower (nearly full trucks) | Lower investment, higher operation time |
| **Logistical Limitation** | Capacity too small | Balanced | Time / scheduling |
| **Key Advantage** | Simple but limited | Full coverage + optimal filling | More trucks |





