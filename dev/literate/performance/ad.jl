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
using ValueHistories: MVHistory
using RestrictedBoltzmannMachines: BinaryRBM, initialize!, cd!, cdad!
nothing #hide

# Setup

Float = Float32
epochs = 100
batchsize = 128
nothing #hide

# Load MNIST

train_x, train_y = MLDatasets.MNIST.traindata()
train_x = Array{Float}(train_x[:, :, train_y .== 0] .≥ 0.5)
nothing #hide

# Train using explicit gradients

rbm_∂s = BinaryRBM(Float, (28,28), 128)
@time cd!(rbm_∂s, train_x) # warm-up run so as to not consider pre-compilation times
initialize!(rbm_∂s, train_x)
history_∂s = MVHistory()
time_0 = time()
@time cd!(
    rbm_∂s, train_x; epochs, batchsize,
    callback = function(@nospecialize(args...); @nospecialize(kw...))
        push!(history_∂s, :t, time() - time_0)
    end
)
nothing #hide

# Train using Zygote gradients

rbm_ad = BinaryRBM(Float, (28,28), 128)
@time cdad!(rbm_ad, train_x) # warm-up run so as to not consider pre-compilation times
initialize!(rbm_ad, train_x)
history_ad = MVHistory()
time_0 = time()
@time cdad!(
    rbm_ad, train_x; epochs, batchsize,
    callback = function(@nospecialize(args...); @nospecialize(kw...))
        push!(history_ad, :t, time() - time_0)
    end
)
nothing #hide

# Compare timings

fig = Makie.Figure(resolution=(600, 400))
ax = Makie.Axis(fig[1,1], xlabel="batch", ylabel="seconds")
Makie.lines!(ax, get(history_∂s, :t)..., label="manual")
Makie.lines!(ax, get(history_ad, :t)..., label="zygote")
Makie.axislegend(ax, position=:rt)
fig
