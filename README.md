## Activate env
```
conda activate /pscratch/sd/j/jwl50/interpolants-torch/.env
```

## Interpolation
### Analytical targets:
```
python -m src.experiments.interpolation.simple_fcns.abs_1d
python -m src.experiments.interpolation.simple_fcns.sine_1d
python -m src.experiments.interpolation.simple_fcns.sine_2d
```
### PDE solutions: TODO
```
python -m src.experiments.interpolation.pde_solns.adv_1d
python -m src.experiments.interpolation.pde_solns.burgers_1d
python -m src.experiments.interpolation.pde_solns.allen_cahn
python -m src.experiments.interpolation.pde_solns.navier_stokes_2d
```

## ODEs: TODO
```
```

## Simple PDEs: TODO
```
adv_1d
poisson_2d
heat_2d
wave_1d
```

## Benchmark PDEs: TODO
```
burgers_1d
allen_cahn
navier_stokes_2d
```

TODO:
- Models
  - [x] ND polynomial interpolant
  - [ ] ND rational interpolant
- Experiments
  - [ ] Interpolation
  - [ ] ODEs and simple PDEs
  - [ ] Benchmark PDEs
