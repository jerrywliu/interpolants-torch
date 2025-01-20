## Activate env
```
conda activate /pscratch/sd/j/jwl50/interpolants-torch/.env
```

## Interpolation
### Analytical targets:
```
python -m src.experiments.interpolation.simple_fcns.abs_1d
python -m src.experiments.interpolation.simple_fcns.sine_1d
python -m src.experiments.interpolation.simple_fcns.logistic_1d
```
### PDE solutions:
```
python -m src.experiments.interpolation.pde_solns.advection
python -m src.experiments.interpolation.pde_solns.reaction
python -m src.experiments.interpolation.pde_solns.wave
```

## ODEs: TODO
```
```

## Simple PDEs:
```
python -m src.experiments.pdes.simple.advection
python -m src.experiments.pdes.simple.reaction
python -m src.experiments.pdes.simple.wave
```

## Benchmark PDEs: TODO Allen-Cahn and Navier-Stokes
```
python -m src.experiments.pdes.benchmarks.burgers
python -m src.experiments.pdes.benchmarks.allen_cahn
python -m src.experiments.pdes.benchmarks.ns_2d
```

TODO:
- Models
  - [x] ND polynomial interpolant
  - [x] 1D rational interpolant
  - [ ] ND rational interpolant
- Experiments
  - [x] Interpolation
  - [ ] ODEs
  - [x] Simple PDEs
  - [ ] Benchmark PDEs
