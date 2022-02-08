# RestrictedBoltzmannMachines.jl Documentation

A Julia package to train and simulate Restricted Boltzmann Machines.
The package is registered.
Install it with:

```julia
import Pkg
Pkg.add("RestrictedBoltzmannMachines")
```

The source code is hosted on Github.

<https://github.com/cossio/RestrictedBoltzmannMachines.jl>

This package doesn't export any symbols.
It can be imported like this:

```julia
import RestrictedBoltzmannMachines as RBMs
```

to avoid typing a long name everytime.

Most of the functions have a helpful docstring.
See [Reference](@ref) section.

See also the Examples listed on the menu on the left side bar to understand how the package works as a whole.

Training info is printed to the debug logger, and are hidden by default.
To enable them, set:

```julia
ENV["JULIA_DEBUG"] = "RestrictedBoltzmannMachines"
```

See <https://docs.julialang.org/en/v1/stdlib/Logging/> for more sophisticated approaches.