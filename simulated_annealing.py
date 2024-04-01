import math
import random
import numpy as np

class simulate_annealing:
    def __init__(self, lower_bound = -1, upper_bound = 1, input_size = 1):
        self.lower_bound = lower_bound
        self.upper_bound  = upper_bound
        self.input_size = input_size

    def simulated_annealing(self, func, initial_solution, temperature, cooling_rate, iterations):
        current_solution = initial_solution
        current_energy = func(current_solution)

        best_solution = current_solution
        best_energy = current_energy

        for _ in range(iterations):
            neighbor = self.generate_neighbor(current_solution)
            neighbor_energy = func(neighbor)

            if neighbor_energy < current_energy or random.random() < self.acceptance_probability(current_energy, neighbor_energy, temperature):
                current_solution = neighbor
                current_energy = neighbor_energy

                if neighbor_energy < best_energy:
                    best_solution = neighbor
                    best_energy = neighbor_energy

            temperature = temperature * cooling_rate

        return best_solution, best_energy

    def generate_neighbor(self, solution):
        perturbation = [random.uniform(self.lower_bound, self.upper_bound) for _ in range(self.input_size)]
        neighbor = solution + np.array(perturbation)
        return neighbor

    def acceptance_probability(self, current_energy, neighbor_energy, temperature):
        if neighbor_energy < current_energy:
            return 1.0
        return math.exp((current_energy - neighbor_energy) / temperature)


if __name__ == "__main__":
    initial_solution = 3.0
    initial_temperature = 100.0
    cooling_rate = 0.999
    iterations = 1000

    def simple_function(x):
        return -np.exp(-x**2/100)*(np.sin(13*x - x**4)**5)*(np.sin(1 - 3*x**2)**2)
    
    simannealing = simulate_annealing()

    best_solution, best_energy = simannealing.simulated_annealing(simple_function, initial_solution, initial_temperature, cooling_rate, iterations)

    print("Best Solution:", best_solution)
    print("Minimum Energy:", best_energy)