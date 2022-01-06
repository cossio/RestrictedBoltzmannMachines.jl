# RestrictedBoltzmannMachines Julia package

[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/cossio/RestrictedBoltzmannMachines.jl/blob/master/LICENSE.md)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://cossio.github.io/RestrictedBoltzmannMachines.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://cossio.github.io/RestrictedBoltzmannMachines.jl/dev)
![](https://github.com/cossio/RestrictedBoltzmannMachines.jl/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/cossio/RestrictedBoltzmannMachines.jl/branch/master/graph/badge.svg?token=O5P8LQTVF3)](https://codecov.io/gh/cossio/RestrictedBoltzmannMachines.jl)
![GitHub repo size](https://img.shields.io/github/repo-size/cossio/RestrictedBoltzmannMachines.jl)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/cossio/RestrictedBoltzmannMachines.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

Train and sample [Restricted Boltzmann Machines](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine) in Julia.
See the [Documentation](https://cossio.github.io/RestrictedBoltzmannMachines.jl/stable) for details.

## Installation

This package is registered.
Install with:

```julia
using Pkg
Pkg.add("RestrictedBoltzmannMachines")
```

Alternatively,

```julia
pkg> add RestrictedBoltzmannMachines
```

from the Pkg REPL.

This package does not export any symbols.
Since the name `RestrictedBoltzmannMachines` is verbose, I recommend to import it as:

```julia
import RestrictedBoltzmannMachines as RBMs
```

## Related packages

Other packages implementing Restricted Boltzmann Machines in Julia:

- https://github.com/dfdx/Boltzmann.jl
- https://github.com/stefan-m-lenz/BoltzmannMachines.jl

Here are some differences:

- **Genericity:** In contrast to those packages, here we adopt a generic approach that allows defining any kind of unit potentials (such as ReLU or dReLU units, which are not included in those two packages), and use them as visible or hidden layers of the RBM.
- **Easily extensible:** This also means the package is easily extensible to other layer types, or to modified training procedures. See `src/train/` directory for exploration of training schemes (CD, PCD, centering, etc.). Other packages tend to support only CD or PCD.
- **Shallow**: On the other hand, those two repos support deep Boltzmann machines, which we do not.

For a Python 3 implementation see https://github.com/jertubiana/PGM, which originates the `dReLU` layer and showcases applications to protein and neural data.

## ToDo

Some features that should be implemented in the future:

- AIS estimation of the partition function.
- Unbiased contrastive divergence sampling algorithm.