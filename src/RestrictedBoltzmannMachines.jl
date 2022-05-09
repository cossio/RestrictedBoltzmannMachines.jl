module RestrictedBoltzmannMachines

import Flux
import Zygote
import ChainRulesCore

using Base: front, tail
using Random: AbstractRNG, GLOBAL_RNG, randexp, randn!, rand!, shuffle!
using LinearAlgebra: Diagonal, logdet, I, dot, norm
using Statistics: mean, std
using LogExpFunctions: softmax, logsumexp, log1pexp, logistic, logaddexp, logsubexp
using SpecialFunctions: erf, erfcx, logerfcx
using FillArrays: Fill, Zeros, Ones, Trues, Falses
using Flux: ADAM, Descent

include("util/util.jl")
include("util/onehot.jl")
include("util/linalg.jl")
include("util/truncated_normal.jl")
include("util/minibatches.jl")

include("layers/abstractlayer.jl")
include("layers/binary.jl")
include("layers/spin.jl")
include("layers/potts.jl")
include("layers/gaussian.jl")
include("layers/relu.jl")
include("layers/drelu.jl")
include("layers/prelu.jl")
include("layers/xrelu.jl")
include("layers/common.jl")

include("rbm.jl")

include("rbms/hopfield.jl")
include("rbms/binary.jl")

include("gauge/zerosum.jl")
include("gauge/rescale_hidden.jl")

include("pseudolikelihood.jl")
include("partition.jl")
include("ais.jl")

include("centered_gradient.jl")

include("train/initialization.jl")
include("train/cd.jl")
include("train/cdad.jl")
include("train/pcd.jl")
include("train/fast.jl")
include("train/rdm.jl")
include("train/optim.jl")

include("regularize.jl")

include("metropolis.jl")

end # module
