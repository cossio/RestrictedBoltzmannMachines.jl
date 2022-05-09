#=
# Moment-matching conditions

The stationarity conditions for ML learning of the RBM imply certain moment-matching
conditions between the data and the RBM distribution.
However, due to biased nature of the PCD approximation, these conditions might not hold
exactly in practice.
=#

import MKL # faster on Intel CPUs
import Makie, CairoMakie
import MLDatasets
import Flux
using Statistics: mean, cor
using LinearAlgebra: norm
using RestrictedBoltzmannMachines: RBM, Binary, transfer_sample, free_energy
using RestrictedBoltzmannMachines: sample_v_from_v, sample_h_from_v, initialize!, pcd!

# load MNIST dataset

train_x = MLDatasets.MNIST(split=:train).features .> 0.5; # binarize
train_y = MLDatasets.MNIST(split=:train).targets;
train_x = train_x[:, :, train_y .== 0]; # only zeros for speed

# floating type we will use

Float = Float32

# compute training epochs from desired number of parameter updates

function train_nepochs(;
    nsamples::Int, # number observations in the data
    nupdates::Int, # desired number of parameter updates
    batchsize::Int # size of each mini-batch
)
    return ceil(Int, nupdates * batchsize / nsamples)
end

# initialize RBM

nhidden = 64
batchsize = 64
nupdates = 10000
epochs = train_nepochs(; nsamples = size(train_x, 3), nupdates, batchsize)
rbm = RBM(Binary(Float,28,28), Binary(Float,nhidden), zeros(Float,28,28,nhidden))
initialize!(rbm, train_x);
@time pcd!(rbm, train_x; epochs, batchsize);

# Generate samples, in v-space and h-space.

nsteps = 2000
nsamples = 1000
v_model = falses(28, 28, nsamples, nsteps);
v_model[:,:,:,1] .= transfer_sample(rbm.visible, falses(28,28,nsamples));
for t in 2:nsteps
    v_model[:, :, :, t] .= sample_v_from_v(rbm, v_model[:, :, :, t - 1])
end
h_data = sample_h_from_v(rbm, train_x);
h_model = sample_h_from_v(rbm, v_model[:,:,:,end]);

v_data_center = train_x .- mean(train_x; dims=3);
h_data_center = h_data .- mean(h_data; dims=2);
vh_data = reshape(v_data_center, 28*28, :) * h_data_center' / size(v_data_center, 3);

v_model_center = v_model[:,:,:,end] .- mean(v_model[:,:,:,end]; dims=3);
h_model_center = h_model .- mean(h_model; dims=2);
vh_model = reshape(v_model_center, 28*28, :) * h_model_center' / size(v_model_center, 3);

# Plot moment-matching conditions

# <v>_data vs. <v>_model

Makie.scatter(
    vec(mean(train_x; dims=3)),
    vec(mean(v_model[:,:,:,end]; dims=3))
)
Makie.xlims!(0,1)
Makie.ylims!(0,1)
Makie.current_figure()

# <h>_data vs. <h>_model

Makie.scatter(
    vec(mean(h_data; dims=2)),
    vec(mean(h_model; dims=2)),
)
Makie.xlims!(0,1)
Makie.ylims!(0,1)
Makie.current_figure()

# <vh>_data vs. <vh>_model

Makie.scatter(vec(vh_data), vec(vh_model))
Makie.xlims!(0,1)
Makie.ylims!(0,1)
Makie.current_figure()

# Correlation

cor(vec(vh_data), vec(vh_model))

# Verify convergence of sampling

Makie.lines(vec(mean(free_energy(rbm, v_model[:,:,1:100,:]); dims=1)))

# Plot average sampled digit

Makie.heatmap(mean(v_model[:,:,:,end]; dims=3)[:,:,1])

# Plot average data digit

Makie.heatmap(mean(train_x; dims=3)[:,:,1])

# Difference

Makie.heatmap(mean(v_model[:,:,:,end]; dims=3)[:,:,1] - mean(train_x; dims=3)[:,:,1])

# Plot weights  of maximum intensity.

𝒫 = sortperm([norm(rbm.w[:,:,m]) for m in 1:nhidden]; rev=true);

fig = Makie.Figure()
for i = 1:7
    ax = Makie.Axis(fig[1,i], width=50, height=50)
    Makie.heatmap!(ax, rbm.w[:, :, 𝒫[i]])
    Makie.hidedecorations!(ax)
    Makie.hidespines!(ax)
end
Makie.resize_to_layout!(fig)
fig
