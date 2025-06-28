module RestrictedBoltzmannMachines

import ChainRulesCore
import LinearAlgebra
using Base: front, tail
using EllipsisNotation: (..)
using FillArrays: Fill, Falses, Trues, Zeros, Ones
using LinearAlgebra: Diagonal, dot, I, logdet, norm
using LogExpFunctions: log1pexp, logaddexp, logistic, logsubexp, logsumexp, softmax
using Optimisers: AbstractRule, Adam, setup, update!
using Random: AbstractRNG, default_rng, rand!, randexp, randn!, randperm, shuffle!
using SpecialFunctions: erf, erfcx, logerfcx
using Statistics: mean, std

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
include("layers/nsReLU.jl")
include("layers/pottsgumbel.jl")
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

include("standardized.jl")
include("centered.jl")

function cpu end
function gpu end
function save_rbm end
function load_rbm end

end # module
