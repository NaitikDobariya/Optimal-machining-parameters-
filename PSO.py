import numpy as np

def pso(function, num_particles, num_dimensions, num_iterations, min_bound, max_bound, inertia_weight, cognitive_weight, social_weight):
    # Initialize particles with random positions and velocities within bounds
    particles = np.random.uniform(min_bound, max_bound, size=(num_particles, num_dimensions))
    velocities = np.zeros((num_particles, num_dimensions))
    personal_best_positions = particles.copy()
    personal_best_values = np.array([function(p) for p in personal_best_positions])
    global_best_index = np.argmin(personal_best_values)
    global_best_position = personal_best_positions[global_best_index]

    for _ in range(num_iterations):
        for i in range(num_particles):
            # Update particle velocity
            r1, r2 = np.random.rand(), np.random.rand()
            cognitive_component = cognitive_weight * r1 * (personal_best_positions[i] - particles[i])
            social_component = social_weight * r2 * (global_best_position - particles[i])
            velocities[i] = inertia_weight * velocities[i] + cognitive_component + social_component

            # Update particle position
            particles[i] += velocities[i]

            # Clip particle position to stay within bounds
            particles[i] = np.clip(particles[i], min_bound, max_bound)

            # Evaluate the objective function at the new position
            value = function(particles[i])

            # Update personal best if necessary
            if value < personal_best_values[i]:
                personal_best_values[i] = value
                personal_best_positions[i] = particles[i]

            # Update global best if necessary
            if value < personal_best_values[global_best_index]:
                global_best_index = i
                global_best_position = particles[i]

    return global_best_position, personal_best_positions

if __name__ == '__main__':
    num_particles = 50
    num_dimensions = 1
    num_iterations = 100
    min_bound = -10
    max_bound = 10
    inertia_weight = 0.7
    cognitive_weight = 1.5
    social_weight = 1.5

    def function(x):
        # return (x[0] - 1.22) ** 2 + (x[1] - 2.45) ** 2
        # return -np.exp(-x**2/100)*(np.sin(13*x - x**4)**5)*(np.sin(1 - 3*x**2)**2)
        return 4*x**4 - 17*x**3 + 12*x**2 - 16*x + 17

    global_best, personal_best = pso(function, num_particles, num_dimensions, num_iterations, min_bound, max_bound, inertia_weight, cognitive_weight, social_weight)

    print("Global Best Solution:", global_best)
    print("Value at Global Best:", function(global_best))