# DatstatEatOptimizationPython
Implements Erika Antonette Tan's Optimization Algorithm into Python

- Enriquez, E. A. T., Mendoza, R. G., & Velasco, A. C. T. (2022). Philippine eagle optimization algorithm. IEEE Access, 10, 29089-29120. Retrieved from https://ieeexplore.ieee.org/abstract/document/9732449.

# Example usage:
```
if __name__ == "__main__":
    # Example: minimize Sphere function on [-5,5]^2, retaining your variable naming.
    D = 2

    def ObjectiveFunction(x: np.ndarray) -> float:
        return float(np.sum(x * x))

    SpaceXMax = np.array([5.0, 5.0])
    SpaceXMin = np.array([-5.0, -5.0])

    # Quick demo budgets (defaults would be IES=20*D^2, IFS=10*D^2, MFE=10000*D)
    result = PhilippineEagleAlgorithm(
        D=D,
        f=ObjectiveFunction,
        SpaceXMax=SpaceXMax,
        SpaceXMin=SpaceXMin,
        IES=200,
        IFS=60,
        MFE=5000,
        Seed=42
    )

    print("Best Fitness:", result.FBest)
    print("Best Solution:", result.XBest)
    print("Evaluations Used:", result.EvalsUsed)
```
