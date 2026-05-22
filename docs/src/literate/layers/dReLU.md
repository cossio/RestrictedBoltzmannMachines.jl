```@meta
EditURL = "dReLU.jl"
```

# [dReLU layer](@id drelu_layer)

The dReLU (double Rectified Linear Unit) layer defines continuous hidden units
with values in ``\mathbb{R}``. Unlike the standard ReLU, it can model
asymmetric distributions by using separate parameters for the positive
and negative parts of the distribution.
This parameterization was introduced by J. Tubiana et al.,
["Learning protein constitutive motifs from sequence data"](https://elifesciences.org/articles/39397).

The potential function is:

```math
U(h) = \frac{\gamma^+}{2} (h^+)^2 + \theta^+ h^+ + \frac{\gamma^-}{2} (h^-)^2 + \theta^- h^-
```

where ``h^+ = \max(0, h)`` and ``h^- = \min(0, h)``.
The parameters ``\gamma^+, \gamma^-`` control the curvature (precision) of
the positive and negative sides independently, while ``\theta^+, \theta^-``
control the location.

In this example we visualize the distribution of dReLU units
for different parameter combinations, showing how the asymmetric
parameterization allows for a rich variety of distribution shapes.

First load the required packages.

````@example dReLU
import RestrictedBoltzmannMachines as RBMs
import Makie
import CairoMakie
using Statistics
nothing #hide
````

Initialize a dReLU layer spanning a grid of parameter values.

````@example dReLU
θps = [-3.0; 3.0]
θns = [-3.0; 3.0]
γps = [0.5; 1.0]
γns = [0.5; 1.0]
layer = RBMs.dReLU(;
    θp = [θp for θp in θps, θn in θns, γp in γps, γn in γns],
    θn = [θn for θp in θps, θn in θns, γp in γps, γn in γns],
    γp = [γp for θp in θps, θn in θns, γp in γps, γn in γns],
    γn = [γn for θp in θps, θn in θns, γp in γps, γn in γns]
)
nothing #hide
````

Sample from the layer (with zero input from the other layer).

````@example dReLU
data = RBMs.sample_from_inputs(layer, zeros(size(layer)..., 10^6))
nothing #hide
````

Each subplot corresponds to a different ``(\theta^+, \theta^-)`` combination.
Within each subplot, different curves show different ``(\gamma^+, \gamma^-)``
combinations, illustrating how the curvature parameters shape the distribution.

````@example dReLU
fig = Makie.Figure(resolution=(1000, 700))
xs = repeat(reshape(range(minimum(data), maximum(data), 100), 1,1,1,1,100), size(layer)...)
ps = exp.(-RBMs.cgfs(layer) .- RBMs.energies(layer, xs))
for (iθp, θp) in enumerate(θps), (iθn, θn) in enumerate(θns)
    ax = Makie.Axis(fig[iθp,iθn], title="θ⁺=$θp, θ⁻=$θn", xlabel="h", ylabel="P(h)")
    for (iγp, γp) in enumerate(γps), (iγn, γn) in enumerate(γns)
        Makie.hist!(ax, data[iθp, iθn, iγp, iγn, :], normalization=:pdf, bins=30, label="γ⁺=$γp, γ⁻=$γn")
        Makie.lines!(ax, xs[iθp, iθn, iγp, iγn, :], ps[iθp, iθn, iγp, iγn, :], linewidth=2)
    end
    if iθp == iθn == 1
        Makie.axislegend(ax)
    end
end
fig
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

