#=
# Gaussian layer

In the following example we look at what the Gaussian layer hidden units look like,
for different parameter values.

First load some packages.
=#

import RestrictedBoltzmannMachines as RBMs
using CairoMakie, Statistics
nothing #hide

#=
Now initialize our Gaussian layer, with unit parameters spanning an interesting range.
=#

θs = [-5; 5]
γs = [1; 2]
layer = RBMs.Gaussian(; θ=[θ for θ in θs, γ in γs], γ=[γ for θ in θs, γ in γs])
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
ps = exp.(RBMs.cgfs(layer) .- RBMs.energies(layer, xs))
for (iθ, θ) in enumerate(θs), (iγ, γ) in enumerate(γs)
    hist!(ax, data[iθ, iγ, :], normalization=:pdf, label="θ=$θ, γ=$γ")
    lines!(xs[iθ, iγ, :], ps[iθ, iγ, :], linewidth=2)
end
axislegend(ax)
fig
