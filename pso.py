import numpy as np
from fitness import fitness, multi_objective_fitness

class PSO:
    def __init__(
        self,
        n_particles=30,
        iterations=100,
        price_min=5,
        price_max=30,  # 🔹 UPDATED
        w=0.7,
        c1=1.5,
        c2=1.5,
        multi_objective=False,
        alpha=0.1
    ):
        self.n_particles = n_particles
        self.iterations = iterations
        self.price_min = price_min
        self.price_max = price_max
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.multi_objective = multi_objective
        self.alpha = alpha

        self.positions = np.random.uniform(price_min, price_max, n_particles)
        self.velocities = np.random.uniform(-1, 1, n_particles)

        self.pbest_positions = self.positions.copy()
        self.pbest_scores = self.evaluate(self.positions)

        self.gbest_position = self.pbest_positions[np.argmin(self.pbest_scores)]
        self.gbest_score = min(self.pbest_scores)

    def evaluate(self, positions):
        if self.multi_objective:
            return np.array([
                multi_objective_fitness(p, self.alpha) for p in positions
            ])
        return np.array([fitness(p) for p in positions])

    def optimize(self):
        history = []

        for _ in range(self.iterations):
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(), np.random.rand()

                self.velocities[i] = (
                    self.w * self.velocities[i]
                    + self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                    + self.c2 * r2 * (self.gbest_position - self.positions[i])
                )

                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(
                    self.positions[i], self.price_min, self.price_max
                )

                score = self.evaluate([self.positions[i]])[0]

                if score < self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_positions[i] = self.positions[i]

            self.gbest_position = self.pbest_positions[np.argmin(self.pbest_scores)]
            self.gbest_score = min(self.pbest_scores)

            history.append(-self.gbest_score)

        return self.gbest_position, -self.gbest_score, history
