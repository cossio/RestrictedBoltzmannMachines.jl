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
Now load the full dataset.
=#

train_x, train_y = MLDatasets.MNIST.traindata()
tests_x, tests_y = MLDatasets.MNIST.testdata()
nothing #hide

#=
`train_x`, `tests_x` contain the digit images, while
`train_y`, `tests_y` contain the labels.
We will train an RBM with binary (0,1) visible and hidden units.
Therefore we binarize the data first.
In addition, we restrict our attention to `0,1` digits only,
so that training and so on are faster.
=#

selected_digits = (0, 1)
train_x = train_x[:, :, train_y .∈ Ref(selected_digits)] .≥ 0.5
tests_x = tests_x[:, :, tests_y .∈ Ref(selected_digits)] .≥ 0.5
train_y = train_y[train_y .∈ Ref(selected_digits)]
tests_y = tests_y[tests_y .∈ Ref(selected_digits)]
train_nsamples = length(train_y)
tests_nsamples = length(tests_y)
(train_nsamples, tests_nsamples)

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
    heatmap!(ax, train_x[:,:, rand(1:train_nsamples)])
end
fig

#=
Initialize an RBM with 100 hidden units.
It is recommended to initialize the weights as random normals with zero mean and
variance `= 1/(number of visible units)`.
See [Glorot & Bengio 2010](http://proceedings.mlr.press/v9/glorot10a).

Notice how we pass the `Float` type, to set the parameter type of the layers and weights
in the RBM.
=#

rbm = RBMs.RBM(RBMs.Binary(Float,28,28), RBMs.Binary(Float,200), randn(Float,28,28,200)/28)
nothing #hide

#=
Initially, the RBM assigns a poor pseudolikelihood to the data.
=#

RBMs.log_pseudolikelihood(rbm, train_x) |> mean

#

RBMs.log_pseudolikelihood(rbm, tests_x) |> mean

#

#=
Incidentally, let us see how long it takes to evaluate the pseudolikelihood on the full
dataset.
=#

@elapsed RBMs.log_pseudolikelihood(rbm, train_x) # pre-compiled by the calls above

#=
This is the cost you pay when training by tracking the pseudolikelihood.
The pseudolikelihood is computed on the full dataset every epoch.
So if this time is too high compared to the computational time of training on an epoch,
we should disable tracking the pseudolikelihood.
=#

#=
Now we train the RBM on the data.
This returns a [MVHistory](https://github.com/JuliaML/ValueHistories.jl) object
containing things like the pseudo-likelihood of the data during training.
We print here the time spent in the training as a rough benchmark.
=#

history = RBMs.train!(
    rbm, train_x; epochs=200, batchsize=128,
    optimizer=Flux.ADAM()
)
nothing #hide

#=
After training, the pseudolikelihood score of the data improves significantly.
=#

RBMs.log_pseudolikelihood(rbm, train_x) |> mean

#

RBMs.log_pseudolikelihood(rbm, tests_x) |> mean

#=
Plot of log-pseudolikelihood during learning.
Note that this shows the pseudolikelihood of the train data.
=#
lines(get(history, :lpl)...)

#=
Now let's generate some random RBM samples.
First, we select random data digits to be initial conditions for the Gibbs sampling:
=#

fantasy_x_init = train_x[:, :, rand(1:train_nsamples, 3 * 8)]
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

@elapsed fantasy_x = RBMs.sample_v_from_v(rbm, fantasy_x_init; steps=10000)

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


# # Parameter initialization

#=
If we initialize parameters, in particular matching the single-site statistics,
the model trains better and faster.
=#

rbm = RBMs.RBM(
    RBMs.Binary(Float,28,28),
    RBMs.Binary(Float,200),
    randn(Float,28,28,200)/28
)
history_init = RBMs.train!(
    rbm, train_x; epochs=200, batchsize=128, initialize=true,
    optimizer=Flux.ADAM()
)
nothing #hide

#=
Compare the learning curves, with and without initialization.
=#

fig = Figure(resolution=(800, 300))
ax = Axis(fig[1,1])
lines!(ax, get(history, :lpl)..., label="no init.")
lines!(ax, get(history_init, :lpl)..., label="init.")
axislegend(ax)
fig

#=
Notice how the pseudolikelihood curve grows a bit faster than before.
=#

RBMs.log_pseudolikelihood(rbm, train_x) |> mean

#

RBMs.log_pseudolikelihood(rbm, tests_x) |> mean

#=
Let's look at some samples generated by this RBM.
=#

fantasy_x_init = train_x[:, :, rand(1:train_nsamples, 3 * 8)]
fantasy_x = RBMs.sample_v_from_v(rbm, fantasy_x_init; steps=10000)
fantasy_x_ = reshape(fantasy_x, 28, 28, 3, 8)
fig = Figure(resolution=(800, 300))
for i in 1:3, j in 1:8
    ax = Axis(fig[i,j], yreversed=true)
    hidedecorations!(ax)
    heatmap!(ax, fantasy_x_[:,:,i,j])
end
fig


# # Weight normalization

#=
The authors of https://arxiv.org/abs/1602.07868 introduce weight normalization to boost
learning.
Let's try it here.
=#

rbm = RBMs.RBM(
    RBMs.Binary(Float,28,28),
    RBMs.Binary(Float,200),
    randn(Float,28,28,200)/28
)
history_wnorm = RBMs.train!(
    rbm, train_x; epochs=200, batchsize=128, initialize=true, weight_normalization=true,
    optimizer=Flux.ADAM(0.005)
)
nothing #hide

#=
Let's see what the learning curves look like.
=#

fig = Figure(resolution=(800, 300))
ax = Axis(fig[1,1])
lines!(ax, get(history, :lpl)..., label="no init.")
lines!(ax, get(history_init, :lpl)..., label="init.")
lines!(ax, get(history_wnorm, :lpl)..., label="w. norm.")
axislegend(ax)
fig
