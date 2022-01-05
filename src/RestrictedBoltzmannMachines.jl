module RestrictedBoltzmannMachines
    using Random, Statistics, LinearAlgebra
    import SpecialFunctions, LogExpFunctions, FillArrays
    import Flux, Zygote, ChainRulesCore
    using ValueHistories: MVHistory

    abstract type AbstractLayer{N} end
    const AbstractTensor{N, T<:Real} = AbstractArray{T,N}

    include("util.jl")
    include("minibatches.jl")
    include("onehot.jl")

    include("trunc_norm/truncnorm.jl")
    include("trunc_norm/rejection.jl")

    include("layers/binary.jl")
    include("layers/spin.jl")
    include("layers/potts.jl")
    include("layers/gaussian.jl")
    include("layers/relu.jl")
    include("layers/drelu.jl")
    include("layers/prelu.jl")
    include("layers/xrelu.jl")
    include("layers/layer.jl")

    include("rbm.jl")

    include("train/initialization.jl")
    include("train/cd.jl")
    include("train/pcd.jl")
    include("train/pcd_center.jl")
    include("train/cd_white.jl")
    include("train/cd_norm.jl")
    include("train/optim.jl")
    include("train/regularize.jl")

    include("zerosum.jl")

    include("pseudolikelihood.jl")
    include("partition.jl")

    include("linalg.jl")
end
