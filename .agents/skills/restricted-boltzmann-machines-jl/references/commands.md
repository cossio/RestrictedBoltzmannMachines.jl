# Commands Reference

Read this file when you need the common local commands for this repository.

## Workspace and package

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using RestrictedBoltzmannMachines'
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Tests

```bash
julia --project=test test/runtests.jl
julia --project=test test/partition.jl
julia --project=test test/ais.jl
julia --project=test test/hdf5.jl
```

Use the narrowest relevant test file first when the change is localized.

## Docs

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

`docs/make.jl` generates markdown from `docs/src/literate/*.jl`, builds the
Documenter site, and then removes the generated markdown again.

## Extensions and examples

```bash
julia --project=. -e 'using CUDA, RestrictedBoltzmannMachines'
julia --project=. -e 'using HDF5, RestrictedBoltzmannMachines'
julia --project=repl repl/stdrbm_mnist_cpu.jl
```

Only run the CUDA and HDF5 checks when those optional dependencies are
available in the environment.
