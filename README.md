# RestrictedBoltzmannMachines Julia package

[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/cossio/RestrictedBoltzmannMachines.jl/blob/master/LICENSE.md)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://cossio.github.io/RestrictedBoltzmannMachines.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://cossio.github.io/RestrictedBoltzmannMachines.jl/dev)
![](https://github.com/cossio/RestrictedBoltzmannMachines.jl/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/cossio/RestrictedBoltzmannMachines.jl/branch/master/graph/badge.svg?token=O5P8LQTVF3)](https://codecov.io/gh/cossio/RestrictedBoltzmannMachines.jl)
![GitHub repo size](https://img.shields.io/github/repo-size/cossio/RestrictedBoltzmannMachines.jl)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/cossio/RestrictedBoltzmannMachines.jl)

Restricted Boltzmann Machines in Julia.
Related packages:

https://github.com/dfdx/Boltzmann.jl

https://github.com/stefan-m-lenz/BoltzmannMachines.jl

In contrast to those packages, here we adopt a generic approach that allows defining any kind of unit potentials (such as ReLU units, which are not included in those two packages), and use them as visible or hidden layers of the RBM.
On the other hand, those two repos support deep Boltzmann machines, which we do not.

For a Python 3 implementation checkout https://github.com/jertubiana/PGM, which originates the `dReLU` layer and showcases nice applications to protein and neural data.

This package is registered.
Install with:

```
] add RestrictedBoltzmannMachines
```

For examples see the documentation (check out the badges above).