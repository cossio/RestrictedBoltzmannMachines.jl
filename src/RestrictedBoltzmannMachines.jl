module RestrictedBoltzmannMachines
    using Random, Statistics, LinearAlgebra,
        StatsFuns, SpecialFunctions, Distributions,
        Roots, ProgressMeter, Zygote, Flux, ValueHistories
    using OneHot
    using Base.Broadcast: broadcasted
    using Base: tail, front, OneTo, @propagate_inbounds, @kwdef
    using Random: GLOBAL_RNG
    using Flux: throttle, params, Params, ADAM
    using Zygote: unbroadcast, @adjoint, Numeric, Grads
    import NNlib

    export MVHistory, Data

    include("utils/util.jl")
    include("utils/tensor.jl")
    include("utils/zygote.jl")

    include("truncnorm/truncnorm.jl")
    include("truncnorm/rejection.jl")

    include("layers/layer.jl")
    include("layers/binary.jl")
    include("layers/spin.jl")
    include("layers/potts.jl")
    include("layers/gaussian.jl")
    include("layers/relu.jl")
    include("layers/drelu.jl")
    include("layers/drelu2.jl")

    include("rbm.jl")
    include("dbm.jl")

    include("train/oadam.jl")
    include("train/init.jl")
    include("train/data.jl")
    include("train/cd.jl")
    include("train/regularize.jl")
    include("train/lr_schedules.jl")
    include("train/gauge.jl")
    include("pseudolikelihood.jl")
    include("partition.jl")

    #= So we can refer to the package using RBMs instead of
    the verbose RestrictedBoltzmannMachines =#
    export RBMs
    const RBMs = RestrictedBoltzmannMachines

    #= So I know the tree hash of the version I am using =#
    using Pkg
    const RBMsPath = joinpath(dirname(pathof(RestrictedBoltzmannMachines)), "..")
    const RBMsHash = bytes2hex(Pkg.GitTools.tree_hash(RBMsPath))
    function print_hash()
        println("Find current RBMs commit id by running this on the git repo:")
        println("git log --format=\"%H %T\" | grep $RBMsHash")
    end
end
