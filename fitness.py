import numpy as np

# Single-objective fitness (maximize revenue)
def fitness(price):
    demand = max(0, 500 - 20 * price)
    revenue = price * demand
    return -revenue  # PSO minimizes

# Multi-objective fitness
def multi_objective_fitness(price, alpha=0.1):
    demand = max(0, 500 - 20 * price)
    revenue = price * demand
    affordability = price
    return -revenue + alpha * affordability
