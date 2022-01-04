using Random, Statistics, Test, GLMakie
import RestrictedBoltzmannMachines as RBMs
import MLDatasets

train_x, train_y = MLDatasets.MNIST.traindata()
tests_x, tests_y = MLDatasets.MNIST.testdata()
train_x = train_x[:,:,train_y .== 0] .≥ 0.5
tests_x = tests_x[:,:,tests_y .== 0] .≥ 0.5



rbm = RBMs.RBM(RBMs.Binary(28,28), RBMs.Binary(100), randn(28, 28, 100) / 28);
mean(RBMs.log_pseudolikelihood(rbm, train_x))
RBMs.train!(rbm, train_x[:,:,1:128]; epochs=2, batchsize=64);
@profview RBMs.train!(rbm, train_x; epochs=4, batchsize=128);

batches = RBMs.minibatches([1,2,3]; batchsize=7)

batches
