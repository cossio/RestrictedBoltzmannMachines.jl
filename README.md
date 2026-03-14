# RestrictedBoltzmannMachines.jl

[![Docs (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://cossio.github.io/RestrictedBoltzmannMachines.jl/stable)

A Julia package for training and sampling [Restricted Boltzmann Machines](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine) (RBMs) — a class of probabilistic generative models with a bipartite structure of visible and hidden units. This package supports a wide range of unit types (binary, spin, Potts, Gaussian, ReLU variants), GPU acceleration via CUDA, and advanced techniques like centered and standardized RBMs.

## Installation

This package is registered. Install with:

```julia
import Pkg
Pkg.add("RestrictedBoltzmannMachines")
```

This package does not export any symbols. Since the name is long, we recommend importing it as:

```julia
import RestrictedBoltzmannMachines as RBMs
```

## Quick start

Train a Binary RBM on binarized MNIST digits and generate samples:

```julia
import RestrictedBoltzmannMachines as RBMs
import MLDatasets

# Load and binarize MNIST data (28×28 images)
train_x = Array{Float32}(MLDatasets.MNIST(split=:train)[:].features .≥ 0.5)

# Create a Binary RBM with 400 hidden units and initialize from data
rbm = RBMs.BinaryRBM(Float32, (28, 28), 400)
RBMs.initialize!(rbm, train_x)

# Train with Persistent Contrastive Divergence
RBMs.pcd!(rbm, train_x; iters=10000, batchsize=256)

# Generate new samples via Gibbs sampling
fantasy = RBMs.sample_v_from_v(rbm, train_x[:, :, 1:100]; steps=3000)
```

## Supported layer types

RBMs can be constructed from any combination of the following visible and hidden layer types:

| Layer | Values | Parameters | Description |
|-------|--------|------------|-------------|
| `Binary` | {0, 1} | θ | Binary units |
| `Spin` | {-1, +1} | θ | Spin units |
| `Potts` | one-hot vectors | θ | Categorical units |
| `Gaussian` | ℝ | θ, γ | Gaussian units |
| `ReLU` | [0, ∞) | θ, γ | Rectified linear units |
| `dReLU` | ℝ | θ⁺, θ⁻, γ⁺, γ⁻ | Double ReLU |
| `pReLU` | ℝ | θ, γ, Δ, η | Parametric ReLU |
| `xReLU` | ℝ | θ, γ, Δ, ξ | Extended ReLU |

Construct an RBM with any pair of layer types using `RBM(visible, hidden, weights)`, or use convenience constructors like `BinaryRBM`, `HopfieldRBM`, etc.

## Key functionality

- **Training**: `pcd!` — Persistent Contrastive Divergence with customizable optimizer (via [Optimisers.jl](https://github.com/FluxML/Optimisers.jl)), regularization (L1, L2 on weights/fields), and callbacks.
- **Sampling**: `sample_v_from_v`, `sample_h_from_v`, `sample_v_from_h` — Gibbs sampling; `metropolis` — Metropolis-Hastings sampling at arbitrary temperature.
- **Evaluation**: `free_energy`, `log_pseudolikelihood`, `log_likelihood`, `reconstruction_error`.
- **Partition function**: `log_partition` (exact, for small RBMs), `aise` / `raise` (Annealed Importance Sampling estimates).
- **Initialization**: `initialize!(rbm, data)` — match single-site statistics of the data.
- **Gauge transforms**: `zerosum!`, `rescale_weights!` — impose gauge constraints (useful for Potts layers).

## GPU support (CUDA)

Move an RBM to/from the GPU using `gpu` and `cpu` (requires [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)):

```julia
import CUDA
using RestrictedBoltzmannMachines: BinaryRBM, cpu, gpu

rbm = BinaryRBM(randn(5), randn(3), randn(5, 3))
rbm_gpu = gpu(rbm)       # transfer to GPU
# ... train or sample on GPU ...
rbm_cpu = cpu(rbm_gpu)   # transfer back to CPU
```

See this [Google Colab notebook](https://colab.research.google.com/drive/1lfY5t6m-j8n19EXHLnV-lRBBfJ_jLk8y?usp=sharing) for a full GPU training example.

## Centered and Standardized RBMs

**CenteredRBM** introduces offset parameters that track mean unit activities, improving training stability ([Melchior et al., 2016](https://jmlr.org/papers/v17/14-237.html); [Montavon & Müller, 2012](#references)):

$$E(\mathbf{v},\mathbf{h}) = -\sum_i a_i v_i - \sum_\mu b_\mu h_\mu - \sum_{i\mu} w_{i\mu} (v_i - c_i)(h_\mu - d_\mu)$$

**StandardizedRBM** further adds scaling parameters that track unit standard deviations:

$$E(\mathbf{v},\mathbf{h}) = -\sum_i \theta_i v_i - \sum_\mu \theta_\mu h_\mu - \sum_{i\mu} w_{i\mu} \frac{v_i - \lambda_i}{\sigma_i} \frac{h_\mu - \lambda_\mu}{\sigma_\mu}$$

Both types support all standard RBM operations (training, sampling, evaluation).

## Documentation

Full documentation with API reference and worked examples (MNIST, Metropolis sampling, AIS partition function estimation, layer-specific guides):

**[https://cossio.github.io/RestrictedBoltzmannMachines.jl/stable](https://cossio.github.io/RestrictedBoltzmannMachines.jl/stable)**

## Related packages

- [AdvRBMs.jl](https://github.com/cossio/AdvRBMs.jl) — Adversarially constrained RBMs
- [StackedTempering.jl](https://github.com/2024stacktemperingrbm/StackedTempering.jl) — Stacked tempering for RBMs

## Citation

If you use this package in a publication, please cite:

> Jorge Fernandez-de-Cossio-Diaz, Simona Cocco, and Rémi Monasson. "Disentangling Representations in Restricted Boltzmann Machines without Adversaries." [Physical Review X 13, 021003 (2023)](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.13.021003).

BibTeX is available in [CITATION.bib](https://github.com/cossio/RestrictedBoltzmannMachines.jl/blob/master/CITATION.bib).

## References

- Montavon, G. & Müller, K.-R. "Deep Boltzmann machines and the centering trick." *Neural Networks: Tricks of the Trade*, Springer, 2012, pp. 621–637.
- Melchior, J., Fischer, A. & Wiskott, L. "How to center deep Boltzmann machines." *JMLR* 17(1), 2016, pp. 3387–3447.
