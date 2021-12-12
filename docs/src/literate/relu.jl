#=
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

θs = collect(-2:2:2)
γs = collect(1:1:3)
layer = RBMs.ReLU([θ for θ in θs, γ in γs], [γ for θ in θs, γ in γs])

#=
Now we sample our layer to collect some data.
=#

data = RBMs.sample_from_inputs(layer, zeros(size(layer)..., 10^6))
nothing #hide

#=
Let's plot the resulting histogram of the activations of each unit.
We also overlay the analytical PDF.
=#

fig = Figure(resolution=(500,500))
ax = Axis(fig[1,1])
for (iθ, θ) in enumerate(θs), (iγ, γ) in enumerate(γs)
    hist!(ax, data[iθ, iγ, :], label="θ=$θ, γ=$γ", normalization=:pdf)
    ps = exp.(-RBMs.relu_energy.(θ, γ, xs) - RBMs.relu_cgf(θ, γ))
    lines!(xs, ps)
end
