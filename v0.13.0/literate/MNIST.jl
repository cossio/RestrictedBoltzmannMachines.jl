#=
# MNIST

We begin by importing the required packages.
We load MNIST via the MLDatasets.jl package.
=#

using Statistics: mean
using Random: bitrand
using ValueHistories: MVHistory
import CairoMakie
import MLDatasets
import Flux
import RestrictedBoltzmannMachines as RBMs
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
In addition, we consider only `0,1` digits so that training is faster.
=#

Float = Float32
train_x, train_y = MLDatasets.MNIST.traindata()
train_x = Array{Float}(train_x[:, :, train_y .∈ Ref((0,1))] .≥ 0.5)
nothing #hide

# Let's visualize some random digits.

nrows, ncols = 10, 15
fig = CairoMakie.Figure(resolution=(40ncols, 40nrows))
ax = CairoMakie.Axis(fig[1,1], yreversed=true)
idx = rand(1:size(train_x,3), nrows * ncols) # random indices of digits
digits = reshape(train_x[:,:,idx], 28, 28, ncols, nrows)
CairoMakie.image!(ax, imggrid(digits), colorrange=(1,0))
CairoMakie.hidedecorations!(ax)
CairoMakie.hidespines!(ax)
fig

# Initialize an RBM with 400 hidden units.

rbm = RBMs.BinaryRBM(Float, (28,28), 400)
RBMs.initialize!(rbm, train_x) # match single-site statistics
nothing #hide

#=
Initially, the RBM assigns a poor pseudolikelihood to the data.
=#

@time RBMs.log_pseudolikelihood(rbm, train_x) |> mean

#=
(Incidentally, note how long it takes to evaluate the pseudolikelihood on the full
dataset.)

Now we train the RBM on the data.
This returns a [MVHistory](https://github.com/JuliaML/ValueHistories.jl)
collecting some info during training.
=#

batchsize = 256
optim = Flux.ADAM()
vm = bitrand(28, 28, batchsize) # fantasy chains
history = MVHistory()
@time for epoch in 1:500
    RBMs.pcd!(rbm, train_x; vm, history, batchsize, optim)
    push!(history, :lpl, mean(RBMs.log_pseudolikelihood(rbm, train_x))) # track pseudolikelihood
end
nothing #hide

# After training, the pseudolikelihood score of the data improves significantly.
# Plot of log-pseudolikelihood of trian data during learning.

CairoMakie.lines(get(history, :lpl)..., axis = (; xlabel = "train time", ylabel="pseudolikelihood"))

# Sample digits from the RBM starting from a random condition.

@elapsed fantasy_x = RBMs.sample_v_from_v(rbm, bitrand(28,28,nrows*ncols); steps=10000)

# Plot the sampled digits.

fig = CairoMakie.Figure(resolution=(40ncols, 40nrows))
ax = CairoMakie.Axis(fig[1,1], yreversed=true)
CairoMakie.image!(ax, imggrid(reshape(fantasy_x, 28, 28, ncols, nrows)), colorrange=(1,0))
CairoMakie.hidedecorations!(ax)
CairoMakie.hidespines!(ax)
fig
