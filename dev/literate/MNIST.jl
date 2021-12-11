#=
We begin by importing the required packages.
We load MNIST via the MLDatasets.jl package.
=#

import RestrictedBoltzmannMachines as RBMs
using CairoMakie, Statistics
import MLDatasets, Flux
nothing #hide

#=
Let's visualize some random digits.
=#

fig = Figure(resolution=(800, 300))
for i in 1:3, j in 1:8
    ax = Axis(fig[i,j], yreversed=true)
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

train_x = train_x .≥ 0.5
tests_x = tests_x .≥ 0.5
nothing #hide

#=
The above `train_x` and `tests_x` are `BitArray`s.
Though convenient in terms of memory space, these are very slow in linear algebra.
Since we frequently multiply data configurations times the weights of our RBM,
we want to speed this up.
So we convert to floats, which have much faster matrix multiplies thanks to BLAS.
We will use `Float32` here.
To hit BLAS, this must be consistent with the types we use in the parameters of the RBM
below.
=#

Float = Float32
train_x = Float.(train_x)
tests_x = Float.(tests_x)
train_y = Float.(train_y)
tests_y = Float.(tests_y)
nothing #hide

#=
Plot some examples of the binarized data.
=#

fig = Figure(resolution=(800, 300))
for i in 1:3, j in 1:8
    ax = Axis(fig[i,j], yreversed=true)
    hidedecorations!(ax)
    heatmap!(ax, train_x[:,:, rand(1:60000)])
end
fig

#=
Initialize an RBM with 100 hidden units.
It is recommended to initialize the weights as random normals with zero mean and
standard deviation `= 1/sqrt(number of hidden units)`.
See [Glorot & Bengio 2010](http://proceedings.mlr.press/v9/glorot10a).

Notice how we pass the `Float` type, to set the parameter type of the layers and weights
in the RBM.
=#

rbm = RBMs.RBM(RBMs.Binary(Float,28,28), RBMs.Binary(Float,200), randn(Float,28,28,200)/28)
nothing #hide

#=
Initially, the RBM assigns a poor pseudolikelihood to the data.
=#

@time RBMs.log_pseudolikelihood(rbm, train_x) |> mean

@time RBMs.log_pseudolikelihood(rbm, tests_x) |> mean

#=
Train the RBM on the data.
This returns a [MVHistory](https://github.com/JuliaML/ValueHistories.jl) object
containing things like the pseudo-likelihood of the data during training.
We print here the time spent in the training as a rough benchmark.
=#

history = RBMs.train!(
    rbm, train_x; epochs=100, batchsize=128,
    optimizer=Flux.ADAMW(0.001f0, (0.9f0, 0.999f0), 1f-4)
)
nothing #hide

#=
After training, the pseudolikelihood score of the data improves significantly.
=#

@time RBMs.log_pseudolikelihood(rbm, train_x) |> mean

@time RBMs.log_pseudolikelihood(rbm, tests_x) |> mean

#=
Plot of log-pseudolikelihood during learning.
Note that this shows the pseudolikelihood of the train data.
=#
lines(get(history, :lpl)...)

#=
Now let's generate some random RBM samples.
First, we select random data digits to be initial conditions for the Gibbs sampling:
=#

fantasy_x_init = train_x[:, :, rand(1:60000, 3 * 8)]
nothing #hide

#=
Let's plot the selected digits.
=#

fantasy_x_init_ = reshape(fantasy_x_init, 28, 28, 3, 8)
fig = Figure(resolution=(800, 300))
for i in 1:3, j in 1:8
    ax = Axis(fig[i,j], yreversed=true)
    hidedecorations!(ax)
    heatmap!(ax, fantasy_x_init_[:,:,i,j])
end
fig

#=
Now we do the Gibbs sampling to generate the RBM digits.
=#

@elapsed fantasy_x = RBMs.sample_v_from_v(rbm, fantasy_x_init; steps=1000)

#=
Plot the resulting samples.
=#

fantasy_x_ = reshape(fantasy_x, 28, 28, 3, 8)
fig = Figure(resolution=(800, 300))
for i in 1:3, j in 1:8
    ax = Axis(fig[i,j], yreversed=true)
    hidedecorations!(ax)
    heatmap!(ax, fantasy_x_[:,:,i,j])
end
fig
