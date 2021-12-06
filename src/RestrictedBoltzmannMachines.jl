module RestrictedBoltzmannMachines
    using Random, Statistics, LinearAlgebra
    import SpecialFunctions, LogExpFunctions
    import Flux, Zygote, ChainRulesCore

    using ValueHistories: MVHistory
    export MVHistory

    export Binary, Spin, Potts
    export Gaussian, StdGaussian
    export ReLU, dReLU, pReLU
    export RBM, flip_layers

    export energy, interaction_energy
    export cgf, free_energy
    export inputs_h_to_v, inputs_v_to_h
    export sample_v_from_h, sample_h_from_v
    export sample_v_from_v, sample_h_from_h
    export reconstruction_error
    export init!, init_weights!
    export train!
    export log_likelihood, log_partition
    export log_pseudolikelihood

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

    include("train/init.jl")
    include("train/cd.jl")

    #include("train/regularize.jl")

    #include("train/lr_schedules.jl")
    #include("train/gauge.jl")

    include("pseudolikelihood.jl")
    include("partition.jl")
end
