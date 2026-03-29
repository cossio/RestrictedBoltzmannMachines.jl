#=
# [Gaussian layer](@id gaussian_layer)

The Gaussian layer defines continuous hidden units with values in ``\mathbb{R}``.
The potential function for a Gaussian unit is:

```math
U(x) = \left(\frac{|\gamma|}{2} x - \theta\right) x
```

where ``\theta`` is a location parameter and ``\gamma`` controls the precision
(inverse variance). Conditioned on the input ``I`` from the other layer,
the unit follows a Gaussian distribution with mean ``(\theta + I) / |\gamma|``
and variance ``1 / |\gamma|``.

In this example we visualize the distribution of Gaussian units
for different values of ``\theta`` and ``\gamma``.

First load the required packages.
=#

import RestrictedBoltzmannMachines as RBMs
using CairoMakie, Statistics
nothing #hide

#=
Initialize a Gaussian layer with units spanning different parameter combinations.
=#

θs = [-5; 5]
γs = [1; 2]
layer = RBMs.Gaussian(; θ=[θ for θ in θs, γ in γs], γ=[γ for θ in θs, γ in γs])
nothing #hide

# Sample from the layer (with zero input from the other layer).

data = RBMs.sample_from_inputs(layer, zeros(size(layer)..., 10^6))
nothing #hide

#=
Plot the empirical histogram of the samples alongside the exact analytical PDF.
The close agreement validates the sampling implementation.
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
