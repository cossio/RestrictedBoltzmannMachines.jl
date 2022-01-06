# ## dReLU layer

#=
In this example we look at what the dReLU layer hidden units look like,
for different parameter values.
=#

#=
First load some packages.
=#

import RestrictedBoltzmannMachines as RBMs
using CairoMakie, Statistics
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

data = RBMs.transfer_sample(layer, zeros(size(layer)..., 10^6))
nothing #hide

#=
Let's plot the resulting histogram of the activations of each unit.
We also overlay the analytical PDF.
=#

fig = Figure(resolution=(1000, 1000))
for (iθp, θp) in enumerate(θps), (iθn, θn) in enumerate(θns)
    ax = Axis(fig[iθp,iθn], title="θp=$θp, θn=$θn", xlabel="h", ylabel="P(h)")
    for (iγp, γp) in enumerate(γps), (iγn, γn) in enumerate(γns)
        hist!(ax, data[iθp, iθn, iγp, iγn, :], normalization=:pdf, bins=30)
        xs = range(extrema(data[iθp, iθn, iγp, iγn, :])..., 100)
        ps = exp.(-RBMs.drelu_energy.(θp, θn, γp, γn, xs) .- RBMs.drelu_free(θp, θn, γp, γn))
        lines!(ax, xs, ps, label="γp=$γp, γn=$γn", linewidth=2)
    end
    if iθp == iθn == 1
        axislegend(ax)
    end
end
fig

# ## Gaussian layer

#=
In the following example we look at what the Gaussian layer hidden units look like,
for different parameter values.
=#

#=
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
layer = RBMs.Gaussian([θ for θ in θs, γ in γs], [γ for θ in θs, γ in γs])
nothing #hide

#=
Now we sample our layer to collect some data.
=#

data = RBMs.transfer_sample(layer, zeros(size(layer)..., 10^6))
nothing #hide

#=
Let's plot the resulting histogram of the activations of each unit.
We also overlay the analytical PDF.
=#

fig = Figure(resolution=(500,500))
ax = Axis(fig[1,1])
xs = range(minimum(data), maximum(data), 100)
for (iθ, θ) in enumerate(θs), (iγ, γ) in enumerate(γs)
    hist!(ax, data[iθ, iγ, :], normalization=:pdf)
    ps = exp.(-RBMs.gauss_energy.(θ, γ, xs) .- RBMs.gauss_free(θ, γ))
    lines!(xs, ps, label="θ=$θ, γ=$γ", linewidth=2)
end
axislegend(ax)
fig

# ## ReLU layer

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

θs = [0; 10]
γs = [5; 10]
layer = RBMs.ReLU([θ for θ in θs, γ in γs], [γ for θ in θs, γ in γs])
nothing #hide

#=
Now we sample our layer to collect some data.
=#

data = RBMs.transfer_sample(layer, zeros(size(layer)..., 10^6))
nothing #hide

#=
Let's plot the resulting histogram of the activations of each unit.
We also overlay the analytical PDF.
=#

fig = Figure(resolution=(500,500))
ax = Axis(fig[1,1])
xs = range(minimum(data), maximum(data), 100)
for (iθ, θ) in enumerate(θs), (iγ, γ) in enumerate(γs)
    hist!(ax, data[iθ, iγ, :], normalization=:pdf)
    ps = exp.(-RBMs.relu_energy.(θ, γ, xs) .- RBMs.relu_free(θ, γ))
    lines!(xs, ps, label="θ=$θ, γ=$γ", linewidth=2)
end
axislegend(ax)
fig
