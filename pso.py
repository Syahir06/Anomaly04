def optimize(self):
    revenue_history = []
    fitness_history = []

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

        revenue_history.append(-self.gbest_score)
        fitness_history.append(self.gbest_score)

    return (
        self.gbest_position,
        -self.gbest_score,
        revenue_history,
        fitness_history
    )
