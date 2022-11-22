#=
# MNIST

We begin by importing the required packages.
We load MNIST via the MLDatasets.jl package.
=#

import Makie
import CairoMakie
import MLDatasets
using Statistics: mean, std, var
using Random: bitrand
using ValueHistories: MVHistory, @trace
using RestrictedBoltzmannMachines: BinaryRBM, sample_from_inputs,
    initialize!, log_pseudolikelihood, pcd!, free_energy, sample_v_from_v
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
train_x = MLDatasets.MNIST(split=:train)[:].features
train_y = MLDatasets.MNIST(split=:train)[:].targets
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
Now we train the RBM on the data.
=#

batchsize = 256
iters = 10000
history = MVHistory()
@time pcd!(
    rbm, train_x; iters, batchsize,
    callback = function(; iter, _...)
        if iszero(iter % 100)
            lpl = mean(log_pseudolikelihood(rbm, train_x))
            @trace history iter lpl
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
fantasy_F[:,1] .= free_energy(rbm, fantasy_x)
@time for t in 2:nsteps
    fantasy_x .= sample_v_from_v(rbm, fantasy_x)
    fantasy_F[:,t] .= free_energy(rbm, fantasy_x)
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
