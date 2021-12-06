import RestrictedBoltzmannMachines as RBMs
using CairoMakie
import MLDatasets

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
history = RBMs.train!(rbm, train_x[:,:,1:512]; epochs=10, batchsize=128)
nothing # hide

#=
Plot log-pseudolikelihood during learning.
=#
lines(get(history, :lpl)...)
