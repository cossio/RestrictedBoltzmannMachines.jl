#=
# MNIST

We begin by importing the required packages.
We load MNIST via the MLDatasets.jl package.
=#

using MKL # uses MKL for linear algebra (optional)
nothing #hide

using CairoMakie, Statistics
import MLDatasets, Flux
import RestrictedBoltzmannMachines as RBMs
nothing #hide

#=
Useful function to plot MNIST digits.
=#

"""
    imggrid(A)

Given a four dimensional tensor `A` of size `(width, height, ncols, nrows)`
containing `width x height` images in a grid of `nrows x ncols`, this returns
a matrix of size `(width * ncols, height * nrows)`, that can be plotted in a heatmap
to display all images.
"""
function imggrid(A::AbstractArray{<:Any,4})
    return reshape(permutedims(A, (1,3,2,4)), size(A,1)*size(A,3), size(A,2)*size(A,4))
end

#=
Let's visualize some random digits.
=#

nrows, ncols = 10, 15
fig = Figure(resolution=(40ncols, 40nrows))
ax = Axis(fig[1,1], yreversed=true)
digits = MLDatasets.MNIST.traintensor()
digits = digits[:,:,rand(1:size(digits,3), nrows * ncols)]
digits = reshape(digits, 28, 28, ncols, nrows)
image!(ax, imggrid(digits), colorrange=(1,0))
hidedecorations!(ax)
hidespines!(ax)
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
train_x = Array{Float}(train_x)
tests_x = Array{Float}(tests_x)
nothing #hide

#=
Plot some examples of the binarized data.
=#

nrows, ncols = 10, 15
fig = Figure(resolution=(40ncols, 40nrows))
ax = Axis(fig[1,1], yreversed=true)
digits = reshape(train_x[:, :, rand(1:size(train_x,3), nrows * ncols)], 28, 28, ncols, nrows)
image!(ax, imggrid(digits), colorrange=(1,0))
hidedecorations!(ax)
hidespines!(ax)
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

@time history = RBMs.pcd!(
    rbm, train_x; epochs=200, batchsize=256, optimizer=Flux.ADAM()
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
Notice the abrupt oscillations at the beginning.
Those come from the poor initialization of the RBM.
We will correct this below.

Now let's generate some random RBM samples.
First, we select random data digits to be initial conditions for the Gibbs sampling, and
let's plot them.
=#

nrows, ncols = 10, 15
fantasy_x = train_x[:, :, rand(1:train_nsamples, nrows * ncols)]
fig = Figure(resolution=(40ncols, 40nrows))
ax = Axis(fig[1,1], yreversed=true)
image!(ax, imggrid(reshape(fantasy_x, 28, 28, ncols, nrows)), colorrange=(1,0))
hidedecorations!(ax)
hidespines!(ax)
fig

#=
Now we do the Gibbs sampling to generate the RBM digits.
=#

@elapsed fantasy_x = RBMs.sample_v_from_v(rbm, fantasy_x; steps=10000)

#=
Plot the resulting samples.
=#

fig = Figure(resolution=(40ncols, 40nrows))
ax = Axis(fig[1,1], yreversed=true)
image!(ax, imggrid(reshape(fantasy_x, 28, 28, ncols, nrows)), colorrange=(1,0))
hidedecorations!(ax)
hidespines!(ax)
fig

# ## Parameter initialization

#=
If we initialize parameters, in particular matching the single-site statistics,
the model trains better and faster.
=#

rbm = RBMs.RBM(
    RBMs.Binary(Float,28,28),
    RBMs.Binary(Float,200),
    randn(Float,28,28,200)/28
)
RBMs.initialize!(rbm, train_x) # match single-site statistics
@time history_init = RBMs.pcd!(
    rbm, train_x; epochs=200, batchsize=256, optimizer=Flux.ADAM()
)
nothing #hide

#=
Compare the learning curves, with and without initialization.
=#

fig = Figure(resolution=(800, 300))
ax = Axis(fig[1,1])
lines!(ax, get(history, :lpl)..., label="no init.")
lines!(ax, get(history_init, :lpl)..., label="init.")
axislegend(ax, position=:rb)
fig

#=
Notice how the pseudolikelihood curve grows a bit faster than before and is smoother.
The initial oscillations are gone.
=#

RBMs.log_pseudolikelihood(rbm, train_x) |> mean

#

RBMs.log_pseudolikelihood(rbm, tests_x) |> mean

#=
Let's look at some samples generated by this RBM.
=#

fantasy_x = train_x[:, :, rand(1:train_nsamples, nrows * ncols)]
fantasy_x .= RBMs.sample_v_from_v(rbm, fantasy_x; steps=10000)
fig = Figure(resolution=(40ncols, 40nrows))
ax = Axis(fig[1,1], yreversed=true)
image!(ax, imggrid(reshape(fantasy_x, 28, 28, ncols, nrows)), colorrange=(1,0))
hidedecorations!(ax)
hidespines!(ax)
fig
