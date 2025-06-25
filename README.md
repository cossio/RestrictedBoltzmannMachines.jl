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

## Usage with CUDA

We define two functions, `cpu` and `gpu` (similar to Flux.jl), to move `RBM` to/from the CPU and GPU.

```julia
import CUDA # if you want to use the GPU, need to import this
using RestrictedBoltzmannMachines: BinaryRBM, cpu, gpu

rbm = BinaryRBM(randn(5), randn(3), randn(5,3)) # in CPU

# copy to GPU
rbm_cu = gpu(rbm)

# ... do some things with rbm_cu on the GPU (e.g. training, sampling)

# copy back to CPU
rbm = cpu(rbm_cu)
```

See this [Google Colab notebook](https://colab.research.google.com/drive/1lfY5t6m-j8n19EXHLnV-lRBBfJ_jLk8y?usp=sharing) for a full example of training and sampling an RBM with GPU.

## CenteredRBM

Train and sample centered Restricted Boltzmann machines in Julia. See [Melchior et al] for the definition of *centered*. Consider an RBM with binary units. Then the centered variant has energy defined by:

$$
E(v,h) = -\sum_i a_i v_i - \sum_\mu b_\mu h_\mu - \sum_{i\mu} w_{i\mu} (v_i - c_i) (h_\mu - d_\mu)
$$

with offset parameters $c_i,d_\mu$. Typically $c_i,d_\mu$ are set to approximate the average activities of $v_i$ and $h_\mu$, respectively, as this seems to help training (see [Montavon et al]).

## StandardizedRBM

Train and sample a *standardized* Restricted Boltzmann machine in Julia. This is a generalization of the [Melchior et al, Montavon et al] centered RBMs. The energy is given by:

$$E(\mathbf{v},\mathbf{h}) = - \sum_{i}\theta_{i}v_{i} - \sum_{\mu}\theta_{\mu}h_{\mu} - \sum_{i\mu}w_{i\mu} \frac{v_{i} - \lambda_{i}}{\sigma_{i}}\frac{h_{\mu} - \lambda_{\mu}}{\sigma_{\mu}}$$

with some offset parameters $\lambda_i,\lambda_\mu$ and scaling parameters $\sigma_i,\sigma_\mu$. Usually $\lambda_i,\lambda_\mu$ track the mean activities of visible and hidden units, while $\sigma_i,\sigma_\mu$ track their standard deviations.

## Related packages

Adversarially constrained RBMs:

- https://github.com/cossio/AdvRBMs.jl

Stacked tempering:

- https://github.com/2024stacktemperingrbm/StackedTempering.jl

## References

* Montavon, Grégoire, and Klaus-Robert Müller. "Deep Boltzmann machines and the centering trick." Neural networks: tricks of the trade. Springer, Berlin, Heidelberg, 2012. 621-637.
* Melchior, Jan, Asja Fischer, and Laurenz Wiskott. "How to center deep Boltzmann machines." The Journal of Machine Learning Research 17.1 (2016): 3387-3447.

## Citation

If you use this package in a publication, please cite:

* Jorge Fernandez-de-Cossio-Diaz, Simona Cocco, and Remi Monasson. "Disentangling representations in Restricted Boltzmann Machines without adversaries." [Physical Review X 13, 021003 (2023)](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.13.021003).

Or you can use the included [CITATION.bib](https://github.com/cossio/RestrictedBoltzmannMachines.jl/blob/master/CITATION.bib).
