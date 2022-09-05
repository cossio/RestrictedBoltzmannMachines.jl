# RestrictedBoltzmannMachines Julia package

[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/cossio/RestrictedBoltzmannMachines.jl/blob/master/LICENSE.md)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://cossio.github.io/RestrictedBoltzmannMachines.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://cossio.github.io/RestrictedBoltzmannMachines.jl/dev)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

![](https://github.com/cossio/RestrictedBoltzmannMachines.jl/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/cossio/RestrictedBoltzmannMachines.jl/branch/master/graph/badge.svg?token=O5P8LQTVF3)](https://codecov.io/gh/cossio/RestrictedBoltzmannMachines.jl)
![GitHub repo size](https://img.shields.io/github/repo-size/cossio/RestrictedBoltzmannMachines.jl)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/cossio/RestrictedBoltzmannMachines.jl)

Train and sample [Restricted Boltzmann machines](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine) in Julia.
See the [Documentation](https://cossio.github.io/RestrictedBoltzmannMachines.jl/stable) for details.

## Installation

This package is registered.
Install with:

```julia
import Pkg
Pkg.add("RestrictedBoltzmannMachines")
```

This package does not export any symbols.
Since the name `RestrictedBoltzmannMachines` is verbose, it can be imported as:

```julia
import RestrictedBoltzmannMachines as RBMs
```

## Related packages

Other packages implementing Restricted Boltzmann Machines in Julia:

- https://github.com/dfdx/Boltzmann.jl
- https://github.com/stefan-m-lenz/BoltzmannMachines.jl

## ToDo

- Unbiased contrastive divergence sampling algorithm.

## Compat requirements

`Project.toml` lists the compatibility requirements for the package.

The syntax of `Project.toml` doesn't allow for comments
(see [#42697](https://github.com/JuliaLang/julia/issues/42697)).
Therefore I explain the compat requirements here.

- **julia 1.6:** Current Julia LTS.
- **DiffRules 1.8:** Need [#74](https://github.com/JuliaDiff/DiffRules.jl/pull/74).
See [a19cd5c](https://github.com/cossio/RestrictedBoltzmannMachines.jl/commit/a19cd5cf38162f1991839cb69532480faca98068).
Since now I am starting to use explicit gradients everywhere, this requirement might eventually go away.
- **Flux 0.12.9:** `ExpDecay` with a `start` step option [#1816](https://github.com/FluxML/Flux.jl/pull/1816).
- **Flux 0.13.4:** `Adam` instead of `ADAM` (and others) [v0.13.4](https://github.com/FluxML/Flux.jl/releases/tag/v0.13.4).
- **MLUtils 0.2:** Using `MLUtils.stack`, not sure when it was introduced because they don't keep a Changelog. Julia 1.9 will introduce a `Base.stack` function that I can use in the future, provided it is differentiable.