using CairoMakie
import MLDatasets
import RestrictedBoltzmannMachines as RBMs

#=
First we load the MNIST dataset.
=#
train_x, train_y = MLDatasets.MNIST.traindata();
tests_x, tests_y = MLDatasets.MNIST.testdata();

#=
We will train an RBM with binary (0,1) visible and hidden units.
Therefore we binarize the data first.
=#
train_x = float(train_x .≥ 0.5);
tests_x = float(tests_x .≥ 0.5);
train_y = float(train_y);
tests_y = float(tests_y);

#=
Initialize an RBM with 100 hidden units.
=#
rbm = RBMs.RBM(RBMs.Binary(28,28), RBMs.Binary(100), randn(28, 28, 100) / 28);

#=
Train the RBM on the data.
This returns a MVHistory object (from https://github.com/JuliaML/ValueHistories.jl),
containing things like the pseudo-likelihood of the data during training.
=#
history = RBMs.train!(rbm, train_x; epochs=10, batchsize=128);

#=
Plot log-pseudolikelihood during learning.
=#
lines(get(history, :lpl)...)
