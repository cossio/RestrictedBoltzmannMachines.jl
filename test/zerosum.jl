using Test: @test, @testset
import Random
import LinearAlgebra
import RestrictedBoltzmannMachines as RBMs

@testset "zerosum" begin
    q = 3
    N = (5,2,3)

    layer = RBMs.Potts(randn(q, N...) .+ 1)
    @assert LinearAlgebra.norm(sum(layer.θ; dims=1)) > 1
    RBMs.zerosum!(layer)
    @test LinearAlgebra.norm(sum(layer.θ; dims=1)) < 1e-10

    M = (4,3,2)
    B = (3,1)

    rbm = RBMs.RBM(RBMs.Potts(randn(q, N...) .+ 1), RBMs.Potts(randn(M...) .+ 1), randn(q, N..., M...) .+ 1)
    @assert LinearAlgebra.norm(sum(rbm.w; dims=1)) > 1
    @assert LinearAlgebra.norm(sum(rbm.visible.θ; dims=1)) > 1
    @assert LinearAlgebra.norm(sum(rbm.hidden.θ; dims=1)) > 1
    RBMs.zerosum!(rbm)
    @test LinearAlgebra.norm(sum(rbm.w; dims=1)) < 1e-10
    @test LinearAlgebra.norm(sum(rbm.w; dims=5)) < 1e-10
    @test LinearAlgebra.norm(sum(rbm.visible.θ; dims=1)) < 1e-10
    @test LinearAlgebra.norm(sum(rbm.hidden.θ; dims=1)) < 1e-10
end
