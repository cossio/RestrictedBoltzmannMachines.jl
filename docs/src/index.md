# RestrictedBoltzmannMachines.jl Documentation

A Julia package to train and simulate Restricted Boltzmann Machines.
The package is registered. You can install it with:

```julia
pkg> add RestrictedBoltzmannMachines
```

from the Pkg REPL, or

```julia
julia> using Pkg; Pkg.add("RestrictedBoltzmannMachines")
```

from the Julia REPL.

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
To enable them, we can do:

```julia
ENV["JULIA_DEBUG"] = RBMs
```

but see <https://docs.julialang.org/en/v1/stdlib/Logging/> for more sophisticated approaches.