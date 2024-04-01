# Optimal-machining-parameters

## Optimal Machining Parameters Determination

An ML-based method to determine optimal machining parameters! This project presents a method for determining optimal machining parameters to minimize surface roughness in machining operations. While the method itself is not an original discovery, this project focuses on implementing and validating it using machine learning and optimization techniques.

### About the Project

The project begins by collecting data from real experiments conducted during machining operations, where the quality metric to be optimized is surface roughness. Leveraging this dataset, the project trains a machine learning algorithm, specifically a neural network and a support vector regressor (SVR), to predict the surface roughness based on the input machining parameters.

Once trained, the machine learning models serve as surrogate functions, enabling the use of various optimization algorithms to determine the optimal machining parameters that minimize surface roughness. The project implements three optimization algorithms—gradient descent, particle swarm optimization (PSO), and simulated annealing— from scratch to find the minima or optimal points of the surrogate functions.

### Trained Models and Results

The project includes trained models using both neural network and SVR approaches. Below are the predicted vs real value plots for each:

- Trained Neural Network:
  ![bc70d262-b11f-428a-825e-199107f90f35](https://github.com/NaitikDobariya/Optimal-machining-parameters-/assets/113834773/991b5227-f179-4ee6-86c4-31c4b54caa0d)

- Trained Support Vector Regressor:
  ![67873d82-fa86-4da3-854d-ade5f2afe25b](https://github.com/NaitikDobariya/Optimal-machining-parameters-/assets/113834773/cfeefc61-3116-40c8-a9db-f6a351b1380e)


Based on the comparison, the support vector regressor exhibited superior performance and was selected for further optimization.

### Implemented Optimization Algorithms

The project provides implementations of the following optimization algorithms:

- Gradient Descent
- Particle Swarm Optimization (PSO)
- Simulated Annealing

These algorithms are utilized to find the optimal machining parameters that minimize surface roughness.

### Libraries Used

The project leverages several libraries for implementation, including:

- Pandas
- Scikit-learn (NuSVR, SVR, MinMaxScaler, StandardScaler, MultiOutputRegressor)
- TensorFlow
- Matplotlib
- NumPy
- Pickle

In addition, the project includes custom implementations of optimization algorithms:

- `simulated_annealing`: Simulated Annealing algorithm implementation
- `gradient_descent`: Gradient Descent algorithm implementation
- `PSO`: Particle Swarm Optimization algorithm implementation

Thank you!
