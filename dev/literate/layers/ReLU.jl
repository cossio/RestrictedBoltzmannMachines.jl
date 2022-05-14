#=
# ReLU layer

In this example we look at what the ReLU layer hidden units look like,
for different parameter values.
=#

#=
First load some packages.
=#

import RestrictedBoltzmannMachines as RBMs
using CairoMakie, Statistics
nothing #hide

#=
Now initialize our ReLU layer, with unit parameters spanning an interesting range.
=#

θs = [0; 10]
γs = [5; 10]
layer = RBMs.ReLU([θ for θ in θs, γ in γs], [γ for θ in θs, γ in γs])
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

fig = Figure(resolution=(700,500))
ax = Axis(fig[1,1])
xs = repeat(reshape(range(minimum(data), maximum(data), 100), 1, 1, 100), size(layer)...)
ps = exp.(RBMs.free_energies(layer) .- RBMs.energies(layer, xs))
for (iθ, θ) in enumerate(θs), (iγ, γ) in enumerate(γs)
    hist!(ax, data[iθ, iγ, :], normalization=:pdf, label="θ=$θ, γ=$γ")
    lines!(xs[iθ, iγ, :], ps[iθ, iγ, :], linewidth=2)
end
axislegend(ax)
fig
