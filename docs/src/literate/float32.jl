#=
# Float32 vs Float64

Compare training performance using `Float32` vs. `Float64`.
=#

import MKL
import MLDatasets
import Makie
import CairoMakie
import RestrictedBoltzmannMachines as RBMs
using RestrictedBoltzmannMachines: cd!
using ValueHistories: MVHistory

# Using Float32

train_x, train_y = MLDatasets.MNIST.traindata()
train_x = Array{Float32}(train_x[:, :, train_y .== 0] .≥ 0.5)
rbm = RBMs.BinaryRBM(Float32, (28,28), 128)
RBMs.initialize!(rbm, train_x)
history32 = MVHistory()
time_0 = time()
@time RBMs.cd!(
    rbm, train_x; epochs=100, batchsize=128, steps=1,
    callback = function(@nospecialize(args...); @nospecialize(kw...))
        push!(history32, :t, time() - time_0)
    end
)
nothing #hide

# Using Float64

train_x, train_y = MLDatasets.MNIST.traindata()
train_x = Array{Float64}(train_x[:, :, train_y .== 0] .≥ 0.5)
rbm = RBMs.BinaryRBM(Float64, (28,28), 128)
RBMs.initialize!(rbm, train_x)
history64 = MVHistory()
time_0 = time()
@time RBMs.cd!(
    rbm, train_x; epochs=100, batchsize=128, steps=1,
    callback = function(@nospecialize(args...); @nospecialize(kw...))
        push!(history64, :t, time() - time_0)
    end
)
nothing #hide

# Compare

fig = Makie.Figure(resolution=(600, 400))
ax = Makie.Axis(fig[1,1], xlabel="epoch", ylabel="seconds")
Makie.lines!(ax, get(history32, :t)..., label="32")
Makie.lines!(ax, get(history64, :t)..., label="64")
Makie.ylims!(ax, low=0)
Makie.axislegend(ax, position=:rt)
fig
