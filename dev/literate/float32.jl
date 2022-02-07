#=
# Float32 vs Float64

Compare training performance using `Float32` vs. `Float64`.
=#

import MKL
import MLDatasets
import Makie
import CairoMakie
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

fig = Makie.Figure(resolution=(600, 400))
ax = Makie.Axis(fig[1,1], xlabel="epoch", ylabel="seconds")
Makie.lines!(ax, get(history32, :Δt)..., label="32")
Makie.lines!(ax, get(history64, :Δt)..., label="64")
Makie.axislegend(ax, position=:rt)
fig
