#=
# MKL vs. OpenBLAS

With an Intel CPU, [MKL](https://github.com/JuliaLinearAlgebra/MKL.jl) is generally faster
than OpenBLAS. Let's do a quick comparison.
=#

# Load MNIST

import MLDatasets
Float = Float32
train_x, train_y = MLDatasets.MNIST.traindata()
train_x = Array{Float}(train_x[:, :, train_y .∈ Ref((0,1))] .≥ 0.5)
nothing #hide

# Make  sure we are using OpenBLAS first:

using LinearAlgebra
LinearAlgebra.__init__() # use OpenBLAS
BLAS.get_config()

# Number of BLAS threads:

BLAS.get_num_threads()

# Train RBM using OpenBLAS.

import RestrictedBoltzmannMachines as RBMs
rbm = RBMs.BinaryRBM(Float, (28,28), 128)
RBMs.initialize!(rbm, train_x)
history_openblas = RBMs.cd!(rbm, train_x; epochs=50, batchsize=128, steps=1)
nothing #hide

# Now load MKL.

using MKL
MKL.__init__() # don't need this on a fresh Julia session
BLAS.get_config()

# Number of BLAS threads:

BLAS.get_num_threads()

# Now let's rerun the RBM training.

RBMs.initialize!(rbm, train_x)
history_mkl = RBMs.cd!(rbm, train_x; epochs=50, batchsize=128, steps=1)
nothing #hide

# The epochs should be somewhat faster with MKL.

import CairoMakie
fig = CairoMakie.Figure(resolution=(600, 400))
ax = CairoMakie.Axis(fig[1,1], xlabel="epoch", ylabel="seconds")
CairoMakie.lines!(ax, get(history_openblas, :Δt)..., label="OpenBLAS")
CairoMakie.lines!(ax, get(history_mkl, :Δt)..., label="MKL")
CairoMakie.axislegend(ax, position=:rt)
fig
