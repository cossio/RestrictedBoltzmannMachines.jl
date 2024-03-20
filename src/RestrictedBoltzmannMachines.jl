module RestrictedBoltzmannMachines

import ChainRulesCore
import LinearAlgebra
using Base: front, tail
using Random: AbstractRNG, GLOBAL_RNG, randexp, randn!, rand!, shuffle!, randperm
using LinearAlgebra: Diagonal, logdet, I, dot, norm
using Statistics: mean, std
using EllipsisNotation: (..)
using LogExpFunctions: softmax, logsumexp, log1pexp, logistic, logaddexp, logsubexp
using SpecialFunctions: erf, erfcx, logerfcx
using FillArrays: Fill, Zeros, Ones, Trues, Falses
using Optimisers: AbstractRule, setup, update!, Adam

include("util/util.jl")
include("util/onehot.jl")
include("util/linalg.jl")
include("util/truncated_normal.jl")

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
include("rbms/spin.jl")
include("rbms/gaussian.jl")

include("pseudolikelihood.jl")
include("partition.jl")
include("ais.jl")

include("train/initialization.jl")
include("train/pcd.jl")
include("train/gradient.jl")
include("from_grad.jl")
#include("train/minibatches.jl")
include("train/infinite_minibatches.jl")

include("gauge/zerosum.jl")
include("gauge/rescale_hidden.jl")
include("gauge/shift_fields.jl")

include("regularize.jl")

include("metropolis.jl")

end # module
