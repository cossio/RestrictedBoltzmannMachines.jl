#=
# dReLU layer

In this example we look at what the dReLU layer hidden units look like,
for different parameter values.
=#

#=
First load some packages.
=#

import RestrictedBoltzmannMachines as RBMs
import Makie
import CairoMakie
using Statistics
nothing #hide

#=
Now initialize our dReLU layer, with unit parameters spanning an interesting range.
=#

θps = [-3.0; 3.0]
θns = [-3.0; 3.0]
γps = [0.5; 1.0]
γns = [0.5; 1.0]
layer = RBMs.dReLU(
    [θp for θp in θps, θn in θns, γp in γps, γn in γns],
    [θn for θp in θps, θn in θns, γp in γps, γn in γns],
    [γp for θp in θps, θn in θns, γp in γps, γn in γns],
    [γn for θp in θps, θn in θns, γp in γps, γn in γns]
)
nothing #hide

#=
Now we sample our layer to collect some data.
=#

data = RBMs.sample_from_inputs(layer, zeros(size(layer)..., 10^6))
nothing #hide

#=
Let's plot the resulting histogram of the activations of each unit.
We also overlay the analytical PDF.
=#

fig = Makie.Figure(resolution=(1000, 700))
xs = repeat(reshape(range(minimum(data), maximum(data), 100), 1,1,1,1,100), size(layer)...)
ps = exp.(RBMs.cfgs(layer) .- RBMs.energies(layer, xs))
for (iθp, θp) in enumerate(θps), (iθn, θn) in enumerate(θns)
    ax = Makie.Axis(fig[iθp,iθn], title="θp=$θp, θn=$θn", xlabel="h", ylabel="P(h)")
    for (iγp, γp) in enumerate(γps), (iγn, γn) in enumerate(γns)
        Makie.hist!(ax, data[iθp, iθn, iγp, iγn, :], normalization=:pdf, bins=30, label="γp=$γp, γn=$γn")
        Makie.lines!(ax, xs[iθp, iθn, iγp, iγn, :], ps[iθp, iθn, iγp, iγn, :], linewidth=2)
    end
    if iθp == iθn == 1
        Makie.axislegend(ax)
    end
end
fig
