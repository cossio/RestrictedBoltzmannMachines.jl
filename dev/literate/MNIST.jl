#=
We begin by importing the required packages. We load MNIST via MLDatasets.jl.
Here we also plot some of the first digits.
=#

import RestrictedBoltzmannMachines as RBMs
using CairoMakie
import MLDatasets

fig = Figure(resolution=(500, 500))
for i in 1:5, j in 1:5
    ax = Axis(fig[i,j])
    hidedecorations!(ax)
    heatmap!(ax, first(MLDatasets.MNIST.traindata(5 * (i - 1) + j)))
end
fig

#=
First we load the MNIST dataset.
=#
train_x, train_y = MLDatasets.MNIST.traindata()
tests_x, tests_y = MLDatasets.MNIST.testdata()
nothing #hide

#=
We will train an RBM with binary (0,1) visible and hidden units.
Therefore we binarize the data first.
=#
train_x = float(train_x .≥ 0.5)
tests_x = float(tests_x .≥ 0.5)
train_y = float(train_y)
tests_y = float(tests_y)
nothing #hide

#=
In the previous code block, notice how we converted `train_x`, `train_y`, ..., and so on,
to floats using `float`, which in this case converts to `Float64`.
The RBM we will define below also uses `Float64` to store weights.
This is important if we want to hit blas matrix multiplies, which are much faster than,
*e.g.*, using a `BitArray` to store the data.
Thus be careful that the data and the RBM weights have the same float type.
=#

#=
Plot some examples of the binarized data (same digits as above).
=#
fig = Figure(resolution=(500, 500))
for i in 1:5, j in 1:5
    ax = Axis(fig[i,j])
    hidedecorations!(ax)
    heatmap!(ax, train_x[:,:,5 * (i - 1) + j])
end
fig

#=
Initialize an RBM with 100 hidden units.
It is recommended to initialize the weights as random normals with zero mean and
standard deviation `= 1/sqrt(number of hidden units)`.
See [Glorot & Bengio 2010](http://proceedings.mlr.press/v9/glorot10a).
=#
rbm = RBMs.RBM(RBMs.Binary(28,28), RBMs.Binary(100), randn(28, 28, 100) / 28)
nothing #hide

#=
Train the RBM on the data.
This returns a MVHistory object
(from [ValueHistories.jl](https://github.com/JuliaML/ValueHistories.jl)),
containing things like the pseudo-likelihood of the data during training.
We print here the time spent in the training as a rough benchmark.
=#
history = RBMs.train!(rbm, train_x; epochs=100, batchsize=128)
nothing #hide

#=
Plot log-pseudolikelihood during learning.
=#
lines(get(history, :lpl)...)

#=
Generate some RBM samples.
=#
dream_x = RBMs.sample_v_from_v(rbm, train_x; steps=20);

fig = Figure(resolution=(500, 500))
for i in 1:5, j in 1:5
    ax = Axis(fig[i,j])
    hidedecorations!(ax)
    heatmap!(ax, dream_x[:,:,5 * (i - 1) + j])
end
fig
