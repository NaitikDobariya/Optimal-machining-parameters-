import numpy as np

def gradient(func, variables, delta = 1e-6):
    gradients = np.empty(shape = variables.shape[0])
    for i, var in enumerate(variables):
        d_variables = variables.copy()
        d_variables[i] = var + delta
        d_func = func(d_variables)
        grad = (d_func - func(variables)) / delta
        gradients[i] = grad
    return gradients

def gradient_descent(func, X0, delta, learning_rate = 0.001, num_iterations = 200):
    X = X0
    for i in range(num_iterations):
        gradients = gradient(func, X, delta = delta)
        X = X - learning_rate * gradients

    return X

if __name__ == "__main__":
    initial_solution = np.array([5.0])
    iterations = 1000

    def simple_function(x):
        return (x - 3) ** 2


    best_solution = gradient_descent(simple_function, np.array([initial_solution]), delta = 1e-6, num_iterations = iterations, learning_rate = 0.2)

    print("Best Solution:", best_solution)
        