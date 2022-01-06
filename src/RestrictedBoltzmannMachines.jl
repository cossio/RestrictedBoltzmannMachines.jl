module RestrictedBoltzmannMachines
    using Random, Statistics, LinearAlgebra
    import SpecialFunctions, LogExpFunctions, FillArrays
    import Flux, Zygote, ChainRulesCore
    import LoopVectorization
    using ValueHistories: MVHistory

    include("util.jl")
    include("minibatches.jl")
    include("onehot.jl")
    include("linalg.jl")

    include("trunc_norm/truncnorm.jl")
    include("trunc_norm/rejection.jl")

    include("layers/layer.jl")
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

    include("zerosum.jl")
    include("gradient.jl")
    include("pseudolikelihood.jl")
    include("partition.jl")
    include("ais.jl")

    include("train/initialization.jl")
    include("train/cd.jl")
    include("train/cdad.jl")
    include("train/pcd.jl")
    include("train/pcd_center.jl")
    include("train/cd_white.jl")
    include("train/weight_normalization.jl")
    include("train/optim.jl")
    include("train/regularize.jl")
end
