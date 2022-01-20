#=
If you have an Intel CPU, then [MKL](https://github.com/JuliaLinearAlgebra/MKL.jl)
is generally faster than OpenBLAS.
Since multiplying by the RBM weights is one of the most time consuming operations
in this package, it is advisable to use MKL.
Let's do a quick comparison.
=#

# To be sure we are using OpenBLAS first, we can run:

using LinearAlgebra
LinearAlgebra.__init__() # don't need this on a fresh Julia session
BLAS.get_config()

# Number of BLAS threads:

BLAS.get_num_threads()

# We use Float32 for the test.

Float = Float32

# Now load MNIST

import MLDatasets
train_x, train_y = MLDatasets.MNIST.traindata()
tests_x, tests_y = MLDatasets.MNIST.testdata()
train_x = Array{Float}(train_x .> 0.5)
tests_x = Array{Float}(tests_x .> 0.5)
nothing #hide

# We train an RBM with plain contrastive divergence.
import RestrictedBoltzmannMachines as RBMs
rbm = RBMs.RBM(RBMs.Binary(Float,28,28), RBMs.Binary(Float,128), zeros(Float,28,28,128))
RBMs.initialize!(rbm, train_x)
history_openblas = RBMs.cd!(rbm, train_x; epochs=50, batchsize=128, verbose=true, steps=1)
nothing #hide

# Since we haven't loaded MKL, this first run should have used OpenBLAS.
# You can confirm that by doing:

using LinearAlgebra
BLAS.get_config()

#=
That should print that it's using libopenblas.
Now load MKL.
If you don't have it installed, run first `pkg> add MKL`.
=#

using MKL
MKL.__init__() # don't need this on a fresh Julia session
BLAS.get_config()

# Number of BLAS threads:

BLAS.get_num_threads()

#=
In Julia 1.7 (which I assume you have), that's all you need to do.
If you are on a fresh Julia session where `MKL` was not loaded before, you can skip
the line `MKL.__init__()`.
See <https://julialang.org/blog/2021/11/julia-1.7-highlights/#libblastrampoline_mkljl>.
Now LinearAlgebra routines should forward to MKL (which should be confimed by the output
of `BLAS.get_config()` above).

If for some reason you happen to want to go back to OpenBlAS, you can do
`LinearAlgebra.__init()__`.
And then `MKL.__init__()` to go back to MKL again, and so on.

Now let's rerun the RBM training.
=#

rbm = RBMs.RBM(RBMs.Binary(Float,28,28), RBMs.Binary(Float,128), zeros(Float,28,28,128))
RBMs.initialize!(rbm, train_x)
history_mkl = RBMs.cd!(rbm, train_x; epochs=50, batchsize=128, verbose=true, steps=1)
nothing #hide

#=
You should see that the epochs are faster with MKL.
=#
using CairoMakie
fig = Figure(resolution=(600, 400))
ax = Axis(fig[1,1], xlabel="epoch", ylabel="duration (seconds)")
lines!(ax, get(history_openblas, :Δt)..., label="OpenBLAS")
lines!(ax, get(history_mkl, :Δt)..., label="MKL")
axislegend(ax, position=:rt)
fig
