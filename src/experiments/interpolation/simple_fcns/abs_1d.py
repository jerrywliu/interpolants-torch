from src.experiments.interpolation.simple_fcns.data.abs_1d import Abs1DTarget

# Compare interpolation of abs(x) using different methods:
# 1. Neural network
# 2. Polynomial interpolation
# 3. Barycentric rational interpolation

if __name__ == "__main__":
    target = Abs1DTarget()
