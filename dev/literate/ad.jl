#=
# Gradients with Zygote

It is possible to calculate gradients with Zygote.
Let's compare performance to explicit gradients.
=#

import MKL
import MLDatasets
import Makie
import CairoMakie
import RestrictedBoltzmannMachines as RBMs
nothing #hide

# Setup

Float = Float32
epochs = 100
batchsize = 128
nothing #hide

# Load MNIST

train_x, train_y = MLDatasets.MNIST.traindata()
train_x = Array{Float}(train_x[:, :, train_y .== 4] .≥ 0.5)
nothing #hide

# Train using explicit gradients

rbm_∂s = RBMs.BinaryRBM(Float, (28,28), 128)
RBMs.initialize!(rbm_∂s, train_x)
history_∂s = RBMs.cd!(rbm_∂s, train_x; epochs, batchsize)
nothing #hide

# Train using Zygote gradients

rbm_ad = RBMs.BinaryRBM(Float, (28,28), 128)
RBMs.initialize!(rbm_ad, train_x)
history_ad = RBMs.cdad!(rbm_ad, train_x; epochs, batchsize)
nothing #hide

# Compare timings

fig = Makie.Figure(resolution=(600, 400))
ax = Makie.Axis(fig[1,1], xlabel="epoch", ylabel="seconds")
Makie.lines!(ax, get(history_∂s, :Δt)..., label="manual")
Makie.lines!(ax, get(history_ad, :Δt)..., label="zygote")
Makie.ylims!(ax, low=0)
Makie.axislegend(ax, position=:rt)
fig
