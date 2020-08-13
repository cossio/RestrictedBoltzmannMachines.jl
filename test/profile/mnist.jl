using Random, Test, Flux
using RestrictedBoltzmannMachines
using RestrictedBoltzmannMachines: Data


train_x = reshape(hcat(MNIST.images(:train)...), 28,28,:)
tests_x = reshape(hcat(MNIST.images(:tests)...), 28,28,:)
train_x = float(float(train_x) .> 0.5)
tests_x = float(float(train_x) .> 0.5)

train_y = MNIST.labels(:train)
tests_y = MNIST.labels(:tests)

train_w = ones(length(train_y))
tests_w = ones(length(tests_y))

# initialize RBM
rbm = RBM(Layer(BernoulliUnit(),28,28), Layer(GaussianUnit(), 100))
randn!(rbm.vis.units.θ)
@time log_pseudolikelihood_rand(rbm, train_x)

# reduce data for faster testing
train_x = train_x[5:15, 5:15, 1:400]
train_y = train_y[1:400]
train_w = train_w[1:400]
train_loader = Data((train_x, train_y, train_w), batchsize=16, shuffle=true, partial=false)

# initialize RBM
rbm = RBM(Layer(BernoulliUnit(),11,11), Layer(GaussianUnit(), 30))
init!(rbm, train_x)
# train
@time train!(rbm, train_loader)


@profiler train!(rbm, train_loader)
