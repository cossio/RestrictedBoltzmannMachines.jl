# [Layer Types](@id layer_types)

An RBM is composed of a visible layer and a hidden layer, each of which can be any of the types listed below. The RBM energy function is:

```math
E(\mathbf{v}, \mathbf{h}) = U_v(\mathbf{v}) + U_h(\mathbf{h}) - \mathbf{v}^\top \mathbf{w}\, \mathbf{h}
```

where ``U_v`` and ``U_h`` are the layer potential functions defined below.
Each layer type defines a different family of conditional distributions for its units.

## Discrete layers

### Binary

Units take values in ``\{0, 1\}``. The potential function is:

```math
U(\mathbf{x}) = -\sum_i \theta_i x_i
```

where ``\theta_i`` are external fields. Conditioned on the other layer, each unit follows an independent Bernoulli distribution with probability ``\sigma(\theta_i + I_i)``, where ``\sigma`` is the sigmoid function and ``I_i`` is the input from the other layer.

Constructed with [`Binary`](@ref) or the convenience function [`BinaryRBM`](@ref).

### Spin

Units take values in ``\{-1, +1\}``. The potential function is:

```math
U(\mathbf{s}) = -\sum_i \theta_i s_i
```

Conditioned on the other layer, each unit takes value ``+1`` with probability ``\sigma(2(\theta_i + I_i))``.

Constructed with [`Spin`](@ref) or [`SpinRBM`](@ref).

### Potts

Units are one-hot encoded categorical variables with ``q`` categories. The potential function is:

```math
U(\mathbf{x}) = -\sum_{i,c} \theta_{c,i}\, x_{c,i}
```

Conditioned on the other layer, each unit follows a categorical distribution (softmax over the ``q`` categories).

Constructed with [`Potts`](@ref). A GPU-optimized variant [`PottsGumbel`](@ref) is also available, which uses the Gumbel-softmax trick for sampling.

## Continuous layers

### Gaussian

Units take values in ``\mathbb{R}``. The potential function is:

```math
U(\mathbf{x}) = \sum_i \left(\frac{|\gamma_i|}{2} x_i - \theta_i \right) x_i
```

where ``\theta_i`` is a location parameter and ``\gamma_i`` controls the precision (inverse variance).
Conditioned on the other layer, each unit follows a Gaussian distribution with mean ``(\theta_i + I_i) / |\gamma_i|`` and variance ``1 / |\gamma_i|``.

Constructed with [`Gaussian`](@ref) or [`GaussianRBM`](@ref).

### ReLU

Units take values in ``[0, \infty)``. The potential function is:

```math
U(\mathbf{x}) = \sum_i \left(\frac{|\gamma_i|}{2} x_i - \theta_i\right) x_i
```

for ``x_i \geq 0`` (with ``U = \infty`` for ``x_i < 0``).
This is a truncated Gaussian: conditioned on the other layer, each unit follows a rectified Gaussian distribution.

Constructed with [`ReLU`](@ref).

### dReLU

Double Rectified Linear Units take values in ``\mathbb{R}``. The potential function decomposes into positive and negative parts:

```math
U(\mathbf{x}) = \sum_i \left(\frac{\gamma_i^+}{2} (x_i^+)^2 + \theta_i^+ x_i^+\right) + \left(\frac{\gamma_i^-}{2} (x_i^-)^2 + \theta_i^- x_i^-\right)
```

where ``x^+ = \max(0, x)`` and ``x^- = \min(0, x)``.
This allows asymmetric distributions with different curvatures for positive and negative values.

Constructed with [`dReLU`](@ref).

### pReLU, xReLU

Parametric and extended ReLU variants with additional parameters for more flexible distributions. See [`pReLU`](@ref) and [`xReLU`](@ref).

## Constructing an RBM

You can construct an RBM from any pair of layer types:

```julia
import RestrictedBoltzmannMachines as RBMs

# Generic constructor
visible = RBMs.Binary(; θ = zeros(28, 28))
hidden = RBMs.ReLU(; θ = zeros(400), γ = ones(400))
weights = randn(28 * 28, 400) / 100
rbm = RBMs.RBM(visible, hidden, weights)
```

Or use convenience constructors:

| Constructor | Visible | Hidden |
|-------------|---------|--------|
| [`BinaryRBM`](@ref) | Binary | Binary |
| [`SpinRBM`](@ref) | Spin | Spin |
| [`GaussianRBM`](@ref) | Gaussian | Gaussian |
| [`HopfieldRBM`](@ref) | Spin | Gaussian |
