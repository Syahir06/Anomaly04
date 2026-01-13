import itertools
from pso import PSO

def tune_pso():
    W_VALUES = [0.4, 0.6, 0.8]
    C1_VALUES = [1.0, 1.5, 2.0]
    C2_VALUES = [1.0, 1.5, 2.0]

    best_result = {
        "w": None,
        "c1": None,
        "c2": None,
        "price": 0,
        "revenue": 0
    }

    for w, c1, c2 in itertools.product(W_VALUES, C1_VALUES, C2_VALUES):
        pso = PSO(
            n_particles=30,
            iterations=80,
            w=w,
            c1=c1,
            c2=c2
        )

        price, revenue, _ = pso.optimize()

        if revenue > best_result["revenue"]:
            best_result = {
                "w": w,
                "c1": c1,
                "c2": c2,
                "price": price,
                "revenue": revenue
            }

    return best_result
