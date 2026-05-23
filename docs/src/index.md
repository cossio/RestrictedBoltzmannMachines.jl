```@meta
CurrentModule = RestrictedBoltzmannMachines
```

# RestrictedBoltzmannMachines.jl

A Julia package for training and sampling [Restricted Boltzmann Machines](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine) (RBMs).

## What is an RBM?

A Restricted Boltzmann Machine is a probabilistic generative model consisting of a layer of **visible** units ``\mathbf{v}`` and a layer of **hidden** units ``\mathbf{h}``, connected by a weight matrix ``\mathbf{w}``. The joint probability distribution is defined by the energy function:

```math
E(\mathbf{v}, \mathbf{h}) = E_v(\mathbf{v}) + E_h(\mathbf{h}) - \mathbf{v}^\top \mathbf{w}\, \mathbf{h}
```

where ``E_v`` and ``E_h`` are layer-specific energy functions that depend on the type of units used (binary, Gaussian, ReLU, etc.).

The key property of RBMs is that visible and hidden units are conditionally independent given the other layer, which allows efficient block Gibbs sampling.

## Standardized RBMs

`StandardizedRBM` augments a plain `RBM` with visible/hidden offsets (`offset_v`, `offset_h`) and scales (`scale_v`, `scale_h`), so interactions are computed from standardized activities.

Its energy is:

```math
E(\mathbf{v},\mathbf{h}) = E_v(\mathbf{v}) + E_h(\mathbf{h}) - \hat{\mathbf{v}}^\top \mathbf{w}\, \hat{\mathbf{h}}
```

where ``\hat{v}_i = (v_i - \lambda_i)/\sigma_i`` and ``\hat{h}_\mu = (h_\mu - \lambda_\mu)/\sigma_\mu`` are standardized activities, ``E_v`` and ``E_h`` are the visible/hidden layer energies, ``\lambda`` are offsets, and ``\sigma`` are scales.

This parameterization is gauge-equivalent to a plain RBM with transformed parameters:

```math
\tilde w_{i\mu} = \frac{w_{i\mu}}{\sigma_i\sigma_\mu},\qquad
\tilde \theta_i = \theta_i - \sum_\mu \tilde w_{i\mu}\lambda_\mu,\qquad
\tilde \theta_\mu = \theta_\mu - \sum_i \tilde w_{i\mu}\lambda_i.
```

So both models represent exactly the same ``P(\mathbf{v},\mathbf{h})``. In practice:

- use `standardize(rbm)` to introduce offsets/scales,
- update them with `standardize_visible_from_data!` and `standardize_hidden_from_v!`,
- use `unstandardize(rbm)` to recover an equivalent plain `RBM`.

## Features

- **Flexible layer types**: Binary, Spin, Potts, Gaussian, ReLU, dReLU, pReLU, xReLU (mix and match for visible and hidden).
- **Training**: Persistent Contrastive Divergence ([`pcd!`](@ref)) with customizable optimizers (via [Optimisers.jl](https://github.com/FluxML/Optimisers.jl)), regularization, and callbacks.
- **Sampling**: Gibbs sampling ([`sample_v_from_v`](@ref), [`sample_h_from_v`](@ref)) and Metropolis-Hastings ([`metropolis!`](@ref)) at arbitrary temperature.
- **Evaluation**: [`free_energy`](@ref), [`log_pseudolikelihood`](@ref), [`log_likelihood`](@ref), [`reconstruction_error`](@ref).
- **Partition function**: Exact enumeration ([`log_partition`](@ref)) and Annealed Importance Sampling ([`aise`](@ref), [`raise`](@ref)).
- **GPU support**: Seamless CPU/GPU transfers via [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl).
- **Advanced techniques**: Centered and Standardized RBMs for improved training stability.
- **Gauge transforms**: [`zerosum!`](@ref), [`rescale_weights!`](@ref) for Potts layers and weight normalization.

## Installation

This package is registered. Install with:

```julia
import Pkg
Pkg.add("RestrictedBoltzmannMachines")
```

This package does not export any symbols.
Since the name is long, we recommend importing it as:

```julia
import RestrictedBoltzmannMachines as RBMs
```

## Quick start

```julia
import RestrictedBoltzmannMachines as RBMs
import MLDatasets

# Load and binarize MNIST data
train_x = Array{Float32}(MLDatasets.MNIST(split=:train)[:].features .≥ 0.5)

# Create a Binary RBM with 400 hidden units
rbm = RBMs.BinaryRBM(Float32, (28, 28), 400)
RBMs.initialize!(rbm, train_x)

# Train with Persistent Contrastive Divergence
RBMs.pcd!(rbm, train_x; iters=10000, batchsize=256)

# Generate samples via Gibbs sampling
samples = RBMs.sample_v_from_v(rbm, train_x[:, :, 1:100]; steps=1000)
```

## Training notes

Typical RBM training flow:
1. construct an RBM with appropriate visible/hidden layers,
2. call [`initialize!`](@ref) once on representative data,
3. train with [`pcd!`](@ref), and
4. monitor progress with [`log_pseudolikelihood`](@ref) or [`reconstruction_error`](@ref).

Useful [`pcd!`](@ref) arguments:
- `iters`: number of parameter updates.
- `batchsize`: mini-batch size.
- `steps`: Gibbs steps for fantasy-particle updates each iteration.
- `optim`: optimizer rule from Optimisers.jl.
- `callback`: receives per-iteration state and can be used for logging.
- `l1_weights`, `l2_weights`, `l2_fields`, `l2l1_weights`: regularization knobs.
- `wts`: optional per-sample weights for weighted datasets.

## Documentation guide

- **[Layer Types](@ref layer_types)**: Overview of all supported layer types with their energy functions and parameters.
- **Examples**: Step-by-step tutorials showing how to train an RBM on [MNIST](@ref), estimate the partition function with [Annealed Importance Sampling](@ref annealed_importance_sampling), and sample with [Metropolis-Hastings](@ref metropolis_sampling). Also includes visualizations of individual layer distributions ([Gaussian](@ref gaussian_layer), [ReLU](@ref relu_layer), [dReLU](@ref drelu_layer)).
- **[Reference](@ref)**: Full API reference with docstrings for all functions and types.

## Source code

The source code is hosted on GitHub: <https://github.com/cossio/RestrictedBoltzmannMachines.jl>.

## Citation

If you use this package in a publication, please cite:

> Jorge Fernandez-de-Cossio-Diaz, Simona Cocco, and Rémi Monasson. "Disentangling Representations in Restricted Boltzmann Machines without Adversaries." [Physical Review X 13, 021003 (2023)](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.13.021003).
