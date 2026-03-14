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

### dReLU, pReLU, xReLU

These three layer types represent the **same family of asymmetric piecewise-quadratic distributions**, differing only in parameterization.
They can be converted to each other without loss of information.

The distribution is defined by a potential that allows different curvatures and locations
for positive and negative values of ``x``:

```math
U(x) = \begin{cases}
\frac{\gamma^+}{2} x^2 + \theta^+ x & \text{if } x \geq 0 \\[4pt]
\frac{\gamma^-}{2} x^2 + \theta^- x & \text{if } x < 0
\end{cases}
```

The three types differ in how they parameterize this distribution:

| Type | Parameters | Notes |
|------|-----------|-------|
| [`dReLU`](@ref) | ``\theta^+, \theta^-, \gamma^+, \gamma^-`` | Separate parameters for positive and negative parts. Direct but redundant. |
| [`pReLU`](@ref) | ``\theta, \gamma, \Delta, \eta`` | Shared scale ``\gamma`` with asymmetry ratio ``\eta \in (-1, 1)``. |
| [`xReLU`](@ref) | ``\theta, \gamma, \Delta, \xi`` | Like pReLU but with unbounded ``\xi \in \mathbb{R}`` (related to ``\eta`` by ``\xi = \eta / (1 - |\eta|)``). |

The conversions between parameterizations are given by:

```math
\gamma = \frac{2|\gamma^+|\,|\gamma^-|}{|\gamma^+| + |\gamma^-|}, \qquad
\eta = \frac{|\gamma^-| - |\gamma^+|}{|\gamma^+| + |\gamma^-|}
```

Use whichever parameterization is most convenient; `dReLU` is the most explicit,
while `pReLU` and `xReLU` separate the overall scale from the asymmetry.

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
