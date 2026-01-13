import numpy as np

# fitness.py

def fitness(price):
    """
    Objective:
    - Maximize revenue
    - Ticket price <= RM30
    - Revenue capped at RM250
    """
    price = min(price, 30)

    demand = max(0, 50 - 1.5 * price)  # scaled to fit revenue ≤ 250
    revenue = price * demand

    # Cap revenue to target
    revenue = min(revenue, 250)
    
    return -revenue  # PSO minimizes

def multi_objective_fitness(price, alpha=0.1):
    price = min(price, 30)

    demand = max(0, 50 - 1.5 * price)
    revenue = min(price * demand, 250)

    affordability = price  # lower is better

    return -revenue + alpha * affordability
