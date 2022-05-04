#=
# MNIST

We begin by importing the required packages.
We load MNIST via the MLDatasets.jl package.
=#

import Makie
import CairoMakie
import MLDatasets
import Flux
import RestrictedBoltzmannMachines as RBMs
using Statistics: mean, std, var
using Random: bitrand
using ValueHistories: MVHistory
using RestrictedBoltzmannMachines: visible, BinaryRBM, transfer_sample, minibatch_count
using RestrictedBoltzmannMachines: initialize!, log_pseudolikelihood, pcd!
nothing #hide

# Useful function to plot grids of MNIST digits.

"""
    imggrid(A)

Given a four dimensional tensor `A` of size `(width, height, ncols, nrows)`
containing `width x height` images in a grid of `nrows x ncols`, this returns
a matrix of size `(width * ncols, height * nrows)`, that can be plotted in a heatmap
to display all images.
"""
imggrid(A::AbstractArray{<:Any,4}) =
    reshape(permutedims(A, (1,3,2,4)), size(A,1)*size(A,3), size(A,2)*size(A,4))

#=
Load the MNIST dataset.
We will train an RBM with binary (0,1) visible and hidden units.
Therefore we binarize the data.
In addition, we consider only one kind of digit so that training is faster.
=#

Float = Float32
train_x, train_y = MLDatasets.MNIST.traindata()
train_x = Array{Float}(train_x[:, :, train_y .== 0] .≥ 0.5)
nothing #hide

# Let's visualize some random digits.

nrows, ncols = 10, 15
fig = Makie.Figure(resolution=(40ncols, 40nrows))
ax = Makie.Axis(fig[1,1], yreversed=true)
idx = rand(1:size(train_x,3), nrows * ncols) # random indices of digits
digits = reshape(train_x[:,:,idx], 28, 28, ncols, nrows)
Makie.image!(ax, imggrid(digits), colorrange=(1,0))
Makie.hidedecorations!(ax)
Makie.hidespines!(ax)
fig

# Initialize an RBM with 400 hidden units.

rbm = BinaryRBM(Float, (28,28), 400)
initialize!(rbm, train_x) # match single-site statistics
nothing #hide

#=
Initially, the RBM assigns a poor pseudolikelihood to the data.
=#

println("log(PL) = ", mean(@time log_pseudolikelihood(rbm, train_x)))

#=
(Incidentally, note how long it takes to evaluate the pseudolikelihood on the full
dataset.)

Now we train the RBM on the data.
This returns a [MVHistory](https://github.com/JuliaML/ValueHistories.jl)
collecting some info during training.
=#

batchsize = 256
batchcount = minibatch_count(train_x; batchsize)
epochs = 500
history = MVHistory()
@time pcd!(
    rbm, train_x; epochs, batchsize,
    callback = function(; rbm, epoch, batch_idx, _...)
        if batch_idx == batchcount && epoch % 5 == 0
            push!(history, :lpl, mean(log_pseudolikelihood(rbm, train_x)))
        end
    end
)
nothing #hide

# After training, the pseudolikelihood score of the data improves significantly.
# Plot of log-pseudolikelihood of trian data during learning.

fig = Makie.Figure(resolution=(500,300))
ax = Makie.Axis(fig[1,1], xlabel = "train time", ylabel="pseudolikelihood")
Makie.lines!(ax, get(history, :lpl)...)
fig

# Sample digits from the RBM starting from a random condition.

nsteps = 3000
fantasy_F = zeros(nrows*ncols, nsteps)
fantasy_x = bitrand(28,28,nrows*ncols)
fantasy_F[:,1] .= RBMs.free_energy(rbm, fantasy_x)
@time for t in 2:nsteps
    fantasy_x .= RBMs.sample_v_from_v(rbm, fantasy_x)
    fantasy_F[:,t] .= RBMs.free_energy(rbm, fantasy_x)
end
nothing #hide

# Check equilibration of sampling

fig = Makie.Figure(resolution=(400,300))
ax = Makie.Axis(fig[1,1], xlabel="sampling time", ylabel="free energy")
fantasy_F_μ = vec(mean(fantasy_F; dims=1))
fantasy_F_σ = vec(std(fantasy_F; dims=1))
Makie.band!(ax, 1:nsteps, fantasy_F_μ - fantasy_F_σ/2, fantasy_F_μ + fantasy_F_σ/2)
Makie.lines!(ax, 1:nsteps, fantasy_F_μ)
fig

# Plot the sampled digits.

fig = Makie.Figure(resolution=(40ncols, 40nrows))
ax = Makie.Axis(fig[1,1], yreversed=true)
Makie.image!(ax, imggrid(reshape(fantasy_x, 28, 28, ncols, nrows)), colorrange=(1,0))
Makie.hidedecorations!(ax)
Makie.hidespines!(ax)
fig
