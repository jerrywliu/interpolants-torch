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

## Simple PDEs: advection, reaction, wave
```
python -m src.experiments.pdes.simple.advection --c 40 --n_t 81 --n_x 80 --n_epochs 100000 --method nys_newton --sample_type standard # Gets to machine precision
python -m src.experiments.pdes.simple.reaction --rho 5 --n_t 41 --n_x 41 --n_epochs 100000 --method nys_newton --sample_type standard
python -m src.experiments.pdes.simple.wave --c 2 --beta 5 --n_t 41 --n_x 41 --n_epochs 100000 --method nys_newton --sample_type standard
```

## Benchmark PDEs: Burgers, Allen-Cahn
TODO true solutions for Burgers and Allen-Cahn, NS 2D
```
python -m src.experiments.pdes.benchmarks.burgers --n_t 81 --n_x 81 --n_epochs 100000 --method nys_newton --sample_type standard
python -m src.experiments.pdes.benchmarks.allen_cahn --n_t 81 --n_x 81 --n_epochs 100000 --method nys_newton --sample_type standard
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
