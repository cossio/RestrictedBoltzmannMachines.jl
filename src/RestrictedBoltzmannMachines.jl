module RestrictedBoltzmannMachines
    using Random, Statistics, LinearAlgebra
    using SpecialFunctions, ProgressMeter, Flux, ValueHistories, OneHot
    using Base.Broadcast: broadcasted
    using Flux: params, Params, ADAM
    using LogExpFunctions: logsumexp, logaddexp, log1pexp

    export MVHistory, Data
    export Binary, Spin, Potts
    export Gaussian, StdGaussian
    export ReLU, dReLU, pReLU
    export RBM, flip_layers

    export energy, interaction_energy, free_energy, mean_free_energy, cgf
    export inputs_h_to_v, inputs_v_to_h
    export sample_v_from_h, sample_h_from_v, sample_v_from_v, sample_h_from_h
    export reconstruction_error
    export init!, init_weights!
    export train!, contrastive_divergence

    # export log_likelihood, log_partition, mean_log_likelihood
    # export log_pseudolikelihood, log_pseudolikelihood_rand

    include("util.jl")
    include("minibatches.jl")

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
    #include("pseudolikelihood.jl")
    #include("partition.jl")
end
