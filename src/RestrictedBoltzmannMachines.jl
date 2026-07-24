module RestrictedBoltzmannMachines

import ChainRulesCore
import LinearAlgebra
using Adapt: Adapt, adapt
using EllipsisNotation: (..)
using FillArrays: Falses, Zeros, Ones
using LinearAlgebra: Diagonal, dot, logdet
using LogExpFunctions: log1pexp, logaddexp, logistic, logsubexp, logsumexp, softmax
using Optimisers: AbstractRule, Adam, setup, update!
using Random: AbstractRNG, default_rng, rand!, randexp, randn!, randperm
using SpecialFunctions: erf, erfcx, logerfcx
using Statistics: mean

include("util/util.jl")
include("util/onehot.jl")
include("util/truncated_normal.jl")

include("layers/abstractlayer.jl")
include("layers/binary.jl")
include("layers/spin.jl")
include("layers/pottsgumbel.jl") # declared before potts.jl, which defines the shared statistics
include("layers/potts.jl")
include("layers/gaussian.jl")
include("layers/relu.jl")
include("layers/drelu.jl")
include("layers/prelu.jl")
include("layers/xrelu.jl")
include("layers/nsReLU.jl")
include("layers/common.jl")

include("rbm.jl")

include("rbms/hopfield.jl")
include("rbms/binary.jl")
include("rbms/spin.jl")
include("rbms/gaussian.jl")

include("pseudolikelihood.jl")
include("partition.jl")
include("ais.jl")

include("train/infinite_minibatches.jl")
include("train/initialization.jl")
include("train/train.jl")
include("train/pcd.jl")
include("train/gradient.jl")
include("from_grad.jl")

include("gauge/zerosum.jl")
include("gauge/rescale_hidden.jl")
include("gauge/shift_fields.jl")

include("regularize.jl")

include("metropolis.jl")

include("standardized.jl")
include("centered.jl")
include("offset_rbms.jl")

include("adapt.jl")

function cpu end
function gpu end
function save_rbm end
function load_rbm end

#= Supported public API. The package exports nothing (use
`import RestrictedBoltzmannMachines as RBMs` or `using ...: name`); `public`
marks the names that are stable API without bringing them into scope. Names not
listed here — internal helpers, the layer-implementation contract, the gradient
(`∂`) interface — are implementation details and may change without a breaking
release. =#
public RBM, CenteredRBM, StandardizedRBM
public Binary, Spin, Potts, Gaussian, ReLU, dReLU, pReLU, xReLU, nsReLU, PottsGumbel
public BinaryRBM, SpinRBM, GaussianRBM, HopfieldRBM,
    CenteredBinaryRBM, BinaryStandardizedRBM, SpinStandardizedRBM
public pcd!, initialize!
public log_pseudolikelihood, log_partition, log_likelihood, aise, raise
public energy, free_energy, interaction_energy
public sample_v_from_h, sample_h_from_v, sample_v_from_v, sample_h_from_h
public mean_h_from_v, mean_v_from_h, var_h_from_v, var_v_from_h, mode_h_from_v, mode_v_from_h
public inputs_h_from_v, inputs_v_from_h
public metropolis, metropolis!, cold_metropolis
public center, center!, uncenter, standardize, standardize!, unstandardize
public mirror, zerosum, zerosum!

end # module
