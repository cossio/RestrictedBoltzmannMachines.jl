# RestrictedBoltzmannMachines Julia package

[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/cossio/RestrictedBoltzmannMachines.jl/blob/master/LICENSE.md)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://cossio.github.io/RestrictedBoltzmannMachines.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://cossio.github.io/RestrictedBoltzmannMachines.jl/dev)
![](https://github.com/cossio/RestrictedBoltzmannMachines.jl/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/cossio/RestrictedBoltzmannMachines.jl/branch/master/graph/badge.svg?token=O5P8LQTVF3)](https://codecov.io/gh/cossio/RestrictedBoltzmannMachines.jl)
![GitHub repo size](https://img.shields.io/github/repo-size/cossio/RestrictedBoltzmannMachines.jl)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/cossio/RestrictedBoltzmannMachines.jl)

Train and sample [Restricted Boltzmann machines](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine) in Julia.
See the [Documentation](https://cossio.github.io/RestrictedBoltzmannMachines.jl/stable) for details.

## Installation

This package is registered. Install with:

```julia
import Pkg
Pkg.add("RestrictedBoltzmannMachines")
```

This package does not export any symbols. Since the name `RestrictedBoltzmannMachines` is long, it can be imported as:

```julia
import RestrictedBoltzmannMachines as RBMs
```

## Related packages

Use RBMs on the GPU (CUDA):

- https://github.com/cossio/CudaRBMs.jl

Centered RBMs:

- https://github.com/cossio/CenteredRBMs.jl

Adversarially constrained RBMs:

- https://github.com/cossio/AdvRBMs.jl

## Citation

This code is released as part of the supporting materials of https://arxiv.org/abs/2206.11600. If you use this package in a publication, please cite:

* Jorge Fernandez-de-Cossio-Diaz, Simona Cocco, and Remi Monasson. "Disentangling representations in Restricted Boltzmann Machines without adversaries." Physical Review X 13, 021003 (2023).

Or you can use the included [CITATION.bib](https://github.com/cossio/RestrictedBoltzmannMachines.jl/blob/master/CITATION.bib).