#=
# [ReLU layer](@id relu_layer)

The ReLU (Rectified Linear Unit) layer defines continuous hidden units
with values in ``[0, \infty)``. The potential function is:

```math
U(x) = \left(\frac{|\gamma|}{2} x - \theta\right) x \quad \text{for } x \geq 0
```

with ``U(x) = \infty`` for ``x < 0``, enforcing non-negativity.
This is equivalent to a truncated Gaussian distribution.
Conditioned on the input ``I`` from the other layer, the unit follows
a rectified Gaussian with location ``(\theta + I) / |\gamma|``
and scale ``1 / |\gamma|``.

In this example we visualize the distribution of ReLU units
for different values of ``\theta`` and ``\gamma``.
=#

import RestrictedBoltzmannMachines as RBMs
using CairoMakie, Statistics
nothing #hide

# Initialize a ReLU layer with different parameter combinations.

θs = [0; 10]
γs = [5; 10]
layer = RBMs.ReLU(; θ = [θ for θ in θs, γ in γs], γ = [γ for θ in θs, γ in γs])
nothing #hide

# Sample from the layer (with zero input from the other layer).

data = RBMs.sample_from_inputs(layer, zeros(size(layer)..., 10^6))
nothing #hide

#=
Plot the empirical histogram alongside the analytical PDF.
Note the characteristic spike at ``x = 0`` from the rectification,
and the Gaussian-like tail for positive values.
=#

fig = Figure(resolution=(700,500))
ax = Axis(fig[1,1], xlabel="x", ylabel="P(x)")
xs = repeat(reshape(range(minimum(data), maximum(data), 100), 1, 1, 100), size(layer)...)
ps = exp.(-RBMs.cgfs(layer) .- RBMs.energies(layer, xs))
for (iθ, θ) in enumerate(θs), (iγ, γ) in enumerate(γs)
    hist!(ax, data[iθ, iγ, :], normalization=:pdf, label="θ=$θ, γ=$γ")
    lines!(xs[iθ, iγ, :], ps[iθ, iγ, :], linewidth=2)
end
axislegend(ax)
fig
