using Test: @test, @testset
using LinearAlgebra: norm
using RestrictedBoltzmannMachines: zerosum!, Potts, RBM

@testset "zerosum" begin
    q = 3
    N = (5,2,3)

    layer = Potts(randn(q, N...) .+ 1)
    @assert norm(sum(layer.θ; dims=1)) > 1
    zerosum!(layer)
    @test norm(sum(layer.θ; dims=1)) < 1e-10

    M = (4,3,2)
    B = (3,1)

    rbm = RBM(Potts(randn(q, N...) .+ 1), Potts(randn(M...) .+ 1), randn(q, N..., M...) .+ 1)
    @assert norm(sum(rbm.w; dims=1)) > 1
    @assert norm(sum(rbm.visible.θ; dims=1)) > 1
    @assert norm(sum(rbm.hidden.θ; dims=1)) > 1
    zerosum!(rbm)
    @test norm(sum(rbm.w; dims=1)) < 1e-10
    @test norm(sum(rbm.w; dims=5)) < 1e-10
    @test norm(sum(rbm.visible.θ; dims=1)) < 1e-10
    @test norm(sum(rbm.hidden.θ; dims=1)) < 1e-10
end
