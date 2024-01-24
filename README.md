# DatstatEatOptimizationPython
Implements Erika Antonette Tan's Optimization Algorithm into Python

# Example usage:
```
if __name__ == "__main__":
    # Define your problem dimension D and objective function f here
    D = 2  # Example dimension
    def objective_function(x):
        return np.sum(x ** 2)  # Example objective: minimize the sum of squares
    
    Space_x_max = np.array([5, 5])
    Space_x_min = np.array([-5, -5])

    start_time = time.time()
    fbest, xbest, evals = pheaglealgorithm(D, objective_function, Space_x_max, Space_x_min)
    print("Best Fitness:", fbest)
    print("Best Solution:", xbest)
    print("Evaluations:", evals)
    print("Time taken: {:.2f} seconds".format(time.time() - start_time))
```
