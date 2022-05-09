#=
# Moment-matching conditions

The stationarity conditions for ML learning of the RBM imply certain moment-matching
conditions between the data and the RBM distribution.
=#

import MKL # faster on Intel CPUs
import Makie, CairoMakie
import MLDatasets
import Flux
using RestrictedBoltzmannMachines: RBM, Binary, transfer_sample, free_energy
using RestrictedBoltzmannMachines: sample_v_from_v, sample_h_from_v, initialize!, pcd!
using Statistics: mean

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

# initialize RBM with 100 hidden units
batchsize = 100
nupdates = 1000
epochs = train_nepochs(; nsamples = size(train_x, 3), nupdates, batchsize)
rbm = RBM(Binary(Float,28,28), Binary(Float,64), zeros(Float,28,28,64))
initialize!(rbm, train_x);
optim = Flux.ADADelta()
@time pcd!(rbm, train_x; epochs, batchsize, optim);

# Generate samples, in v-space and h-space.

nsteps = 200
nsamples = 1000
samples_v = falses(28, 28, nsamples, nsteps);
samples_v[:,:,:,1] .= transfer_sample(rbm.visible, falses(28,28,nsamples));
@time for t in 2:nsteps
    samples_v[:, :, :, t] .= sample_v_from_v(rbm, samples_v[:, :, :, t - 1])
end
data_h = sample_h_from_v(rbm, train_x);
samples_h = sample_h_from_v(rbm, samples_v[:,:,:,end]);

# Plot moment-matching conditions

# <v>_data vs. <v>_model

Makie.scatter(
    vec(mean(train_x; dims=3)),
    vec(mean(samples_v[:,:,:,end]; dims=3))
)
Makie.xlims!(0,1)
Makie.ylims!(0,1)
Makie.current_figure()

# <h>_data vs. <h>_model

Makie.scatter(
    vec(mean(data_h; dims=2)),
    vec(mean(samples_h; dims=2)),
)
Makie.xlims!(0,1)
Makie.ylims!(0,1)
Makie.current_figure()

# <vh>_data vs. <vh>_model

Makie.scatter(
    vec(reshape(train_x, 28*28, :) * data_h' / size(train_x, 3)),
    vec(reshape(samples_v[:,:,:,end], 28*28, :) * samples_h' / size(samples_v, 3))
)
Makie.xlims!(0,1)
Makie.ylims!(0,1)
Makie.current_figure()

# Verify convergence of sampling

Makie.lines(vec(mean(free_energy(rbm, samples_v); dims=1)))

# Plot average data digit

Makie.heatmap(mean(samples_v[:,:,:,end]; dims=3)[:,:,1])

# Plot average sampled digit

Makie.heatmap(mean(train_x; dims=3)[:,:,1])

# Difference

Makie.heatmap(mean(samples_v[:,:,:,end]; dims=3)[:,:,1] - mean(train_x; dims=3)[:,:,1])
