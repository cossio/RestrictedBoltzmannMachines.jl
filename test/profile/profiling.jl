using Random, Test, Statistics, Flux
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

# reduce data for faster testing
train_x = train_x[:, :, 1:500]
train_y = train_y[1:500]
train_w = train_w[1:500]
train_loader = Data((train_x, train_y, train_w), batchsize=16, shuffle=true, partial=false)

# initialize RBM
rbm = RBM(Layer(BernoulliUnit(),28,28), Layer(dReLUUnit(),400))
init!(rbm.vis, train_x)
init_weights!(rbm)

# train
using Profile
Profile.init(n=10^7, delay=0.01)

@time train!(rbm, train_loader)
@profiler train!(rbm, train_loader)

train!(rbm, train_loader, PCD(1))
#@descend train!(rbm, train_loader)
first(train_loader)

for d in train_loader
    @assert eltype(d) == Float64
end

using MLDataUtils

train_x, train_y = MNIST.traindata()
train_x = float(train_x .> 0.5)
dl = RandomBatches(train_x, size=64, count=10)

train_loader = Data(train_x_reduced, batchsize=16, shuffle=true, partial=false)
eltype(train_loader)

function do_something(data)
    s = 0.0
    for d in data
        mean(d)
    end
    return s
end

@code_warntype do_something(dl)
@code_warntype do_something(train_loader)
do_something(dl)

for i = 1:10
    println("holamundo")
end


using Flux, MLDatasets
train_x, train_y = MNIST.traindata()
train_loader = Data(train_x, batchsize=64, shuffle=true, partial=false)
eltype(train_loader) # Any

function do_something(data)
    s = 0.0
    for d in data
        s += mean(d)
    end
    return s
end

@code_warntype do_something(train_loader)

first(train_loader)

do_something(train_loader)

@inferred do_something(train_loader)

tensormul_ff(rbm.weights, first(train_loader), Val(ndims(rbm.vis)))
