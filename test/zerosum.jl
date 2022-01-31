using Test, Random, LinearAlgebra, Statistics, DelimitedFiles
import Zygote, Flux, Distributions, SpecialFunctions, LogExpFunctions, QuadGK, NPZ
import RestrictedBoltzmannMachines as RBMs

@testset "zerosum" begin
    q = 3
    N = (5,2,3)

    layer = RBMs.Potts(q, N...)
    randn!(layer.θ)
    layer.θ .+= 1
    @assert norm(sum(layer.θ; dims=1)) > 1
    randn!(layer.θ)
    RBMs.zerosum!(layer)
    @test norm(sum(layer.θ; dims=1)) < 1e-10

    M = (4,3,2)
    B = (3,1)

    rbm = RBMs.RBM(RBMs.Potts(q, N...), RBMs.Potts(M...), randn(q, N..., M...))

    randn!(rbm.visible.θ)
    randn!(rbm.hidden.θ)
    rand!(rbm.hidden.θ)

    rbm.w .+= 1
    rbm.visible.θ .+= 1
    rbm.hidden.θ .+= 1

    @assert norm(sum(rbm.w; dims=1)) > 1
    @assert norm(sum(rbm.visible.θ; dims=1)) > 1
    @assert norm(sum(rbm.hidden.θ; dims=1)) > 1

    RBMs.zerosum!(rbm)

    @test norm(sum(rbm.w; dims=1)) < 1e-10
    @test norm(sum(rbm.w; dims=5)) < 1e-10
    @test norm(sum(rbm.visible.θ; dims=1)) < 1e-10
    @test norm(sum(rbm.hidden.θ; dims=1)) < 1e-10
end
