#=
# Learning rate decay

Effect of decaying the learning rate during training to achieve convergence.
=#

using Statistics: mean, std, var
using Random: bitrand
using LinearAlgebra: dot
using ValueHistories: MVHistory
import Makie
import CairoMakie
import Flux
import MLDatasets
using RestrictedBoltzmannMachines: BinaryRBM, initialize!, pcd!, log_pseudolikelihood, minibatch_count
using RestrictedBoltzmannMachines: free_energy, sample_v_from_v, mean_h_from_v, default_optimizer

#=
Load MNIST dataset. We select only 0 digits and binarize pixel intensities.
=#

Float = Float32
train_x, train_y = MLDatasets.MNIST.traindata()
train_x = Array{Float64}(train_x[:,:, train_y .== 0] .> 0.5)
nsamples = size(train_x)[end]
nothing #hide

# Some hyper-parameters

nh = 100 # number of hidden units
epochs = 1000
batchsize = 128
batchcount = minibatch_count(train_x; batchsize)
nothing #hide

#=
Consider first an RBM that we train without decaying the learning rate.
=#

rbm_nodecay = initialize!(BinaryRBM(Float, (28,28), nh), train_x)
history_nodecay = MVHistory()
@time pcd!(
    rbm_nodecay, train_x; epochs, batchsize,
    optim = default_optimizer(nsamples, batchsize, epochs; decay_after=1),
    callback = function callback(; rbm, epoch, batch_idx, _...)
        if batch_idx == batchcount && epoch % 5 == 0
            push!(history_nodecay, :lpl, mean(log_pseudolikelihood(rbm, train_x)))
        end
    end
)
nothing #hide

#=
Now train an RBM with lr decay after half training (this is the default
behavior).
=#

rbm_decaylr = initialize!(BinaryRBM(Float, (28,28), nh), train_x)
history_decaylr = MVHistory()
@time pcd!(
    rbm_decaylr, train_x; epochs, batchsize,
    callback = function callback(; rbm, epoch, batch_idx, _...)
        if batch_idx == batchcount && epoch % 5 == 0
            push!(history_decaylr, :lpl, mean(log_pseudolikelihood(rbm, train_x)))
        end
    end
)
nothing #hide

#=
Compare the results
=#

fig = Makie.Figure(resolution=(600,400))
ax = Makie.Axis(fig[1,1])
Makie.lines!(ax, get(history_nodecay, :lpl)..., label="normal")
Makie.lines!(ax, get(history_decaylr, :lpl)..., label="decay")
Makie.axislegend(ax, position=:rb)
fig

#=
Check convergence by computing the moment-matching conditions.
First generate MC data from the RBMs.
=#

nsteps = 1000
nsamples = 5000
F_nodecay = zeros(nsamples, nsteps)
F_decaylr = zeros(nsamples, nsteps)
samples_v_nodecay = bitrand(28,28,nsamples)
samples_v_decaylr = bitrand(28,28,nsamples)
F_nodecay[:,1] .= free_energy(rbm_nodecay, samples_v_nodecay)
F_decaylr[:,1] .= free_energy(rbm_decaylr, samples_v_decaylr)
@time for step in 2:nsteps
    samples_v_nodecay .= sample_v_from_v(rbm_nodecay, samples_v_nodecay)
    samples_v_decaylr .= sample_v_from_v(rbm_decaylr, samples_v_decaylr)
    F_nodecay[:,step] .= free_energy(rbm_nodecay, samples_v_nodecay)
    F_decaylr[:,step] .= free_energy(rbm_decaylr, samples_v_decaylr)
end
nothing #hide

# Check equilibration of sampling

# Samples without lr decay

fig = Makie.Figure(resolution=(400,300))
ax = Makie.Axis(fig[1,1])
F_nodecay_μ = vec(mean(F_nodecay; dims=1))
F_nodecay_σ = vec(std(F_nodecay; dims=1))
Makie.band!(ax, 1:nsteps, F_nodecay_μ - F_nodecay_σ/2, F_nodecay_μ + F_nodecay_σ/2)
Makie.lines!(ax, 1:nsteps, F_nodecay_μ, label="no decay")
fig

# Samples with lr decay

fig = Makie.Figure(resolution=(400,300))
ax = Makie.Axis(fig[1,1])
F_decaylr_μ = vec(mean(F_decaylr; dims=1))
F_decaylr_σ = vec(std(F_decaylr; dims=1))
Makie.band!(ax, 1:nsteps, F_decaylr_μ - F_decaylr_σ/2, F_decaylr_μ + F_decaylr_σ/2)
Makie.lines!(ax, 1:nsteps, F_decaylr_μ, label="decay lr")
fig

# Now make the plots. Average digit shapes.

fig = Makie.Figure(resolution=(900, 300))
ax = Makie.Axis(fig[1,1], title="data", yreversed=true)
Makie.heatmap!(ax, mean(train_x, dims=3)[:,:,1])
Makie.hidedecorations!(ax)
ax = Makie.Axis(fig[1,2], title="const. lr", yreversed=true)
Makie.heatmap!(ax, mean(samples_v_nodecay, dims=3)[:,:,1])
Makie.hidedecorations!(ax)
ax = Makie.Axis(fig[1,3], title="lr decay", yreversed=true)
Makie.heatmap!(ax, mean(samples_v_decaylr, dims=3)[:,:,1])
Makie.hidedecorations!(ax)
fig

#=
Moment matching conditions, first row for RBM with constant learning rate,
second row for RBM with learning rate decay.
=#

h_data_nodecay = mean_h_from_v(rbm_nodecay, train_x)
h_data_decaylr = mean_h_from_v(rbm_decaylr, train_x)
h_model_nodecay = mean_h_from_v(rbm_nodecay, samples_v_nodecay)
h_model_decaylr = mean_h_from_v(rbm_decaylr, samples_v_decaylr)
nothing #hide

fig = Makie.Figure(resolution=(900, 600))

ax = Makie.Axis(fig[1,1], xlabel="<v>_data", ylabel="<v>_model", limits=(0,1,0,1))
Makie.scatter!(ax, vec(mean(train_x; dims=3)), vec(mean(samples_v_nodecay; dims=3)))
Makie.abline!(ax, 0, 1; color=:red)

ax = Makie.Axis(fig[1,2], xlabel="<h>_data", ylabel="<h>_model", limits=(0,1,0,1))
Makie.scatter!(ax, vec(mean(h_data_nodecay; dims=2)), vec(mean(h_model_nodecay; dims=2)))
Makie.abline!(ax, 0, 1; color=:red)

ax = Makie.Axis(fig[1,3], xlabel="<vh>_data", ylabel="<vh>_model", limits=(0,1,0,1))
Makie.scatter!(ax,
    vec([dot(train_x[i,j,:], h_data_nodecay[μ,:]) / size(train_x,3) for i=1:28, j=1:28, μ=1:nh]),
    vec([dot(samples_v_nodecay[i,j,:], h_model_nodecay[μ,:]) / size(samples_v_nodecay,3) for i=1:28, j=1:28, μ=1:nh])
)
Makie.abline!(ax, 0, 1; color=:red)

ax = Makie.Axis(fig[2,1], xlabel="<v>_data", ylabel="<v>_model", limits=(0,1,0,1))
Makie.scatter!(ax, vec(mean(train_x; dims=3)), vec(mean(samples_v_decaylr; dims=3)))
Makie.abline!(ax, 0, 1; color=:red)

ax = Makie.Axis(fig[2,2], xlabel="<h>_data", ylabel="<h>_model", limits=(0,1,0,1))
Makie.scatter!(ax, vec(mean(h_data_decaylr; dims=2)), vec(mean(h_model_decaylr; dims=2)))
Makie.abline!(ax, 0, 1; color=:red)

ax = Makie.Axis(fig[2,3], xlabel="<vh>_data", ylabel="<vh>_model", limits=(0,1,0,1))
Makie.scatter!(ax,
    vec([dot(train_x[i,j,:], h_data_decaylr[μ,:]) / size(train_x,3) for i=1:28, j=1:28, μ=1:nh]),
    vec([dot(samples_v_decaylr[i,j,:], h_model_decaylr[μ,:]) / size(samples_v_decaylr,3) for i=1:28, j=1:28, μ=1:nh])
)
Makie.abline!(ax, 0, 1; color=:red)

fig
