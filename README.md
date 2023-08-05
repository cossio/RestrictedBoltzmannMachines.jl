# RestrictedBoltzmannMachines Julia package

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://cossio.github.io/RestrictedBoltzmannMachines.jl/stable)

Train and sample [Restricted Boltzmann machines](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine) in Julia.

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

Centered and standardized RBMs:

- https://github.com/cossio/CenteredRBMs.jl
- https://github.com/cossio/StandardizedRestrictedBoltzmannMachines.jl

Adversarially constrained RBMs:

- https://github.com/cossio/AdvRBMs.jl

Save RBMs to HDF5 files:

- https://github.com/cossio/RestrictedBoltzmannMachinesHDF5.jl

## Citation

If you use this package in a publication, please cite:

* Jorge Fernandez-de-Cossio-Diaz, Simona Cocco, and Remi Monasson. "Disentangling representations in Restricted Boltzmann Machines without adversaries." [Physical Review X 13, 021003 (2023)](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.13.021003).

Or you can use the included [CITATION.bib](https://github.com/cossio/RestrictedBoltzmannMachines.jl/blob/master/CITATION.bib).