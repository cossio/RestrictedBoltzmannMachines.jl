#=
# Learning rate decay

Effect of decaying the learning rate during training to achieve convergence.
=#

using Statistics: mean
using Random: bitrand
using LinearAlgebra: dot
using ValueHistories: MVHistory
import CairoMakie
import Flux
import MLDatasets
import RestrictedBoltzmannMachines as RBMs

#=
Load MNIST dataset. We select only 0 digits and binarize pixel intensities.
=#

Float = Float32
train_x, train_y = MLDatasets.MNIST.traindata()
train_x = Array{Float64}(train_x[:,:, train_y .== 0] .> 0.5)
nothing #hide

#=
Consider first an RBM that we train without decaying the learning rate.
We will train this machine for 300 epochs.
=#

rbm_nodecay = RBMs.BinaryRBM(Float, (28,28), 100)
RBMs.initialize!(rbm_nodecay, train_x)
optim = Flux.ADAM(0.001)
batchsize = 256
vm = bitrand(28, 28, batchsize) # fantasy chains
history_nodecay = MVHistory()
for epoch = 1:300
    RBMs.pcd!(rbm_nodecay, train_x; vm, history=history_nodecay, batchsize, optim)
    push!(history_nodecay, :lpl, mean(RBMs.log_pseudolikelihood(rbm_nodecay, train_x)))
end
nothing #hide

#=
Now train an RBM with 200 normal epochs, followed by 100 epochs where the
learning-rate is cut in half every 10 epochs.
=#

nh = 100
rbm_decaylr = RBMs.BinaryRBM(Float, (28,28), nh)
RBMs.initialize!(rbm_decaylr, train_x)
optim = Flux.ADAM(0.001)
vm = bitrand(28, 28, batchsize)
history_decaylr = MVHistory()
for epoch = 1:200 # first 200 epochs without lr decay
    RBMs.pcd!(rbm_decaylr, train_x; vm, history=history_decaylr, batchsize, optim)
    push!(history_decaylr, :lpl, mean(RBMs.log_pseudolikelihood(rbm_decaylr, train_x)))
end
@info "*** decaying learning rate ****"
for iter = 1:10, epoch = 1:10 # further 10 epochs with decaying lr
    optim.eta = 0.001 / 2^iter
    RBMs.pcd!(rbm_decaylr, train_x; vm, history=history_decaylr, batchsize, optim)
    push!(history_decaylr, :lpl, mean(RBMs.log_pseudolikelihood(rbm_decaylr, train_x)))
end
nothing #hide

#=
Compare the results
=#

fig = CairoMakie.Figure(resolution=(600,400))
ax = CairoMakie.Axis(fig[1,1])
CairoMakie.lines!(ax, get(history_nodecay, :lpl)..., label="normal")
CairoMakie.lines!(ax, get(history_decaylr, :lpl)..., label="decay")
CairoMakie.axislegend(ax, position=:rb)
fig

#=
Check convergence by computing the moment-matching conditions.
First generate MC data from the RBMs.
=#

@time samples_v_nodecay = RBMs.sample_v_from_v(rbm_nodecay, bitrand(28,28,5000); steps=1000)
@time samples_v_decaylr = RBMs.sample_v_from_v(rbm_decaylr, bitrand(28,28,5000); steps=1000)
nothing #hide

#=
Now make the plots. Average digit shapes.
=#

fig = CairoMakie.Figure(resolution=(900, 300))
ax = CairoMakie.Axis(fig[1,1], title="data")
CairoMakie.heatmap!(ax, mean(train_x, dims=3)[:,:,1])
CairoMakie.hidedecorations!(ax)
ax = CairoMakie.Axis(fig[1,2], title="const. lr")
CairoMakie.heatmap!(ax, mean(samples_v_nodecay, dims=3)[:,:,1])
CairoMakie.hidedecorations!(ax)
ax = CairoMakie.Axis(fig[1,3], title="lr decay")
CairoMakie.heatmap!(ax, mean(samples_v_decaylr, dims=3)[:,:,1])
CairoMakie.hidedecorations!(ax)
fig

#=
Moment matching conditions, first for RBM with constant learning rate
=#

h_data_nodecay = RBMs.mean_h_from_v(rbm_nodecay, train_x)
h_data_decaylr = RBMs.mean_h_from_v(rbm_decaylr, train_x)
h_model_nodecay = RBMs.mean_h_from_v(rbm_nodecay, samples_v_nodecay)
h_model_decaylr = RBMs.mean_h_from_v(rbm_decaylr, samples_v_decaylr)
nothing #hide

fig = CairoMakie.Figure(resolution=(900, 600))

ax = CairoMakie.Axis(fig[1,1], xlabel="<v>_data", ylabel="<v>_model", limits=(0,1,0,1))
CairoMakie.scatter!(ax, vec(mean(train_x; dims=3)), vec(mean(samples_v_nodecay; dims=3)))
CairoMakie.abline!(ax, 0, 1; color=:red)

ax = CairoMakie.Axis(fig[1,2], xlabel="<h>_data", ylabel="<h>_model", limits=(0,1,0,1))
CairoMakie.scatter!(ax, vec(mean(h_data_nodecay; dims=2)), vec(mean(h_model_nodecay; dims=2)))
CairoMakie.abline!(ax, 0, 1; color=:red)

ax = CairoMakie.Axis(fig[1,3], xlabel="<vh>_data", ylabel="<vh>_model", limits=(0,1,0,1))
CairoMakie.scatter!(ax,
    vec([dot(train_x[i,j,:], h_data_nodecay[μ,:]) / size(train_x,3) for i=1:28, j=1:28, μ=1:nh]),
    vec([dot(samples_v_nodecay[i,j,:], h_model_nodecay[μ,:]) / size(samples_v_nodecay,3) for i=1:28, j=1:28, μ=1:nh])
)
CairoMakie.abline!(ax, 0, 1; color=:red)

ax = CairoMakie.Axis(fig[2,1], xlabel="<v>_data", ylabel="<v>_model", limits=(0,1,0,1))
CairoMakie.scatter!(ax, vec(mean(train_x; dims=3)), vec(mean(samples_v_decaylr; dims=3)))
CairoMakie.abline!(ax, 0, 1; color=:red)

ax = CairoMakie.Axis(fig[2,2], xlabel="<h>_data", ylabel="<h>_model", limits=(0,1,0,1))
CairoMakie.scatter!(ax, vec(mean(h_data_decaylr; dims=2)), vec(mean(h_model_decaylr; dims=2)))
CairoMakie.abline!(ax, 0, 1; color=:red)

ax = CairoMakie.Axis(fig[2,3], xlabel="<vh>_data", ylabel="<vh>_model", limits=(0,1,0,1))
CairoMakie.scatter!(ax,
    vec([dot(train_x[i,j,:], h_data_decaylr[μ,:]) / size(train_x,3) for i=1:28, j=1:28, μ=1:nh]),
    vec([dot(samples_v_decaylr[i,j,:], h_model_decaylr[μ,:]) / size(samples_v_decaylr,3) for i=1:28, j=1:28, μ=1:nh])
)
CairoMakie.abline!(ax, 0, 1; color=:red)

fig
