module RestrictedBoltzmannMachines
    import Random
    import Statistics
    import LinearAlgebra
    import SpecialFunctions
    import LogExpFunctions
    import FillArrays
    import Flux
    import Zygote
    import ChainRulesCore
    using ValueHistories: MVHistory

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

    include("abstractrbm.jl")
    include("rbm.jl")
    include("special_rbms.jl")

    include("zerosum.jl")
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
end
