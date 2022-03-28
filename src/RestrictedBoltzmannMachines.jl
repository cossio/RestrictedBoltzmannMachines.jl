module RestrictedBoltzmannMachines

import Flux
import Zygote
import ChainRulesCore

using Random: AbstractRNG, GLOBAL_RNG, randexp, randn!, shuffle!
using LinearAlgebra: Diagonal, logdet, I, dot, norm
using Statistics: mean
using LogExpFunctions: softmax, logsumexp, log1pexp, logistic, logaddexp
using SpecialFunctions: erf, erfcx, logerfcx
using ValueHistories: MVHistory
using FillArrays: Fill, Zeros, Ones, Trues, Falses
using Flux: ADAM, Descent

include("util.jl")
include("minibatches.jl")
include("onehot.jl")
include("linalg.jl")
include("truncnorm.jl")

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

# "standardized" layers, with unit scale parameters
include("layers/std/gauss.jl")
include("layers/std/relu.jl")
include("layers/std/xrelu.jl")

include("abstractrbm.jl")
include("rbm.jl")

include("rbms/hopfield.jl")
include("rbms/binary.jl")

include("gauge/zerosum.jl")
include("gauge/rescale_hidden.jl")

include("pseudolikelihood.jl")
include("partition.jl")
include("ais.jl")

include("train/initialization.jl")
include("train/cd.jl")
include("train/cdad.jl")
include("train/pcd.jl")
include("train/bnorm.jl")
include("train/fast.jl")
include("train/rdm.jl")
include("train/optim.jl")
include("train/regularize.jl")

end # module
