import RestrictedBoltzmannMachines as RBMs
using Test: @test, @testset
using Statistics: mean

@testset "initialization Binary" begin
    data = rand(2, 3, 10^6) .≤ 3/4
    rbm = RBMs.RBM(RBMs.Binary(2,3), RBMs.Binary(0), randn(2,3,0))
    RBMs.initialize!(rbm, data)
    @test RBMs.mean_from_inputs(rbm.visible) ≈ reshape(mean(data; dims=3), size(rbm.visible)) rtol=0.01
end

@testset "initialization Spin" begin
    data = Int8.(sign.(rand(2, 3, 10^6) .- 1/4))
    rbm = RBMs.RBM(RBMs.Spin(2,3), RBMs.Binary(0), randn(2,3,0))
    RBMs.initialize!(rbm, data)
    @test RBMs.mean_from_inputs(rbm.visible) ≈ reshape(mean(data; dims=3), size(rbm.visible)) rtol=0.01
end

@testset "initialization Potts" begin
    rbm = RBMs.RBM(RBMs.Potts(3,2,3), RBMs.Binary(0), randn(3,2,3,0))
    data = RBMs.onehot_encode(RBMs.onehot_decode(rand(3,2,3,10^6) .- rand(3,2,3,1)), 1:3)
    @assert all(sum(data; dims=1) .== 1)
    RBMs.initialize!(rbm, data)
    @test RBMs.mean_from_inputs(rbm.visible) ≈ reshape(mean(data; dims=4), size(rbm.visible)) rtol=0.01
end
