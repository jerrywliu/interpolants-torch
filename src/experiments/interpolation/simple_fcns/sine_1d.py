from src.experiments.interpolation.simple_fcns.data.sine_1d import Sine1DTarget

# Compare interpolation of sin(x) using different methods:
# 1. Neural network
# 2. Polynomial interpolation (Chebyshev)
# 3. Polynomial interpolation (Fourier)

if __name__ == "__main__":
    target = Sine1DTarget()
