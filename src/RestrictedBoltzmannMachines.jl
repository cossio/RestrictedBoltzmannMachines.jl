module RestrictedBoltzmannMachines
    using Random, Statistics, LinearAlgebra
    import SpecialFunctions, LogExpFunctions
    import Flux, Zygote, ChainRulesCore
    using ValueHistories: MVHistory

    include("util.jl")
    include("minibatches.jl")
    include("onehot.jl")

    include("truncnorm/truncnorm.jl")
    include("truncnorm/rejection.jl")

    include("layers/binary.jl")
    include("layers/spin.jl")
    include("layers/potts.jl")
    include("layers/gaussian.jl")
    include("layers/relu.jl")
    include("layers/drelu.jl")
    include("layers/prelu.jl")
    include("layers/common.jl")

    include("rbm.jl")

    include("train/initialization.jl")
    include("train/cd.jl")

    #include("train/regularize.jl")

    #include("train/lr_schedules.jl")
    include("train/zerosum.jl")

    include("pseudolikelihood.jl")
    include("partition.jl")
end
