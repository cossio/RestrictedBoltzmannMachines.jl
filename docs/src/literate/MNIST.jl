#=
We begin by importing the required packages.
We load MNIST via the MLDatasets.jl package.
=#

import RestrictedBoltzmannMachines as RBMs
using CairoMakie
import MLDatasets
nothing #hide

#=
Let's visualize some random digits.
=#

fig = Figure(resolution=(700, 300))
for i in 1:3, j in 1:7
    ax = Axis(fig[i,j])
    hidedecorations!(ax)
    heatmap!(ax, MLDatasets.MNIST.traintensor(rand(1:60000)))
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
The RBM we will define below also uses `Float64` to store weights (it's the default).
This is important if we want to hit blas matrix multiplies, which are much faster than,
*e.g.*, using a `BitArray` to store the data.
Thus be careful that the data and the RBM weights have the same float type.
=#

#=
Plot some examples of the binarized data.
=#

fig = Figure(resolution=(700, 300))
for i in 1:3, j in 1:7
    ax = Axis(fig[i,j])
    hidedecorations!(ax)
    heatmap!(ax, train_x[:,:, rand(1:60000)])
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
Plot of log-pseudolikelihood during learning.
=#
lines(get(history, :lpl)...)

#=
Now let's generate some random RBM samples.
First, we select random data digits to be initial conditions for the Gibbs sampling:
=#

fantasy_x_init = train_x[:, :, rand(1:60000, 21)]

#=
Let's plot the selected digits.
=#

fantasy_x_init = reshape(fantasy_x_init, 28, 28, 3, 7)
fig = Figure(resolution=(700, 300))
for i in 1:3, j in 1:7
    ax = Axis(fig[i,j])
    hidedecorations!(ax)
    heatmap!(ax, fantasy_x_init[:,:,i,j])
end
fig

#=
Now we do the Gibbs sampling
=#

@elapsed fantasy_x = RBMs.sample_v_from_v(rbm, fantasy_x_init; steps=100)

#=
Plot the resulting samples
=#

fantasy_x = reshape(fantasy_x, 28, 28, 3, 7)
fig = Figure(resolution=(700, 300))
for i in 1:3, j in 1:7
    ax = Axis(fig[i,j])
    hidedecorations!(ax)
    heatmap!(ax, fantasy_x[:,:,i,j])
end
fig
