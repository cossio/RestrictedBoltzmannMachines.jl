#=
# Float32 vs Float64

Compare training performance using `Float32` vs. `Float64`.
=#

import MKL, MLDatasets
import RestrictedBoltzmannMachines as RBMs

# Using Float32

train_x, train_y = MLDatasets.MNIST.traindata()
train_x = Array{Float32}(train_x[:, :, train_y .∈ Ref((0,1))] .≥ 0.5)
rbm = RBMs.BinaryRBM(Float32, (28,28), 128)
RBMs.initialize!(rbm, train_x)
history32 = RBMs.cd!(rbm, train_x; epochs=100, batchsize=128, steps=1)
nothing #hide

# Using Float64

train_x, train_y = MLDatasets.MNIST.traindata()
train_x = Array{Float64}(train_x[:, :, train_y .∈ Ref((0,1))] .≥ 0.5)
rbm = RBMs.BinaryRBM(Float64, (28,28), 128)
RBMs.initialize!(rbm, train_x)
history64 = RBMs.cd!(rbm, train_x; epochs=100, batchsize=128, steps=1)
nothing #hide

# Compare

import CairoMakie
fig = CairoMakie.Figure(resolution=(600, 400))
ax = CairoMakie.Axis(fig[1,1], xlabel="epoch", ylabel="seconds")
CairoMakie.lines!(ax, get(history32, :Δt)..., label="32")
CairoMakie.lines!(ax, get(history64, :Δt)..., label="64")
CairoMakie.axislegend(ax, position=:rt)
fig
