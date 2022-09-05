using Test: @test, @testset
using Statistics: mean
using RestrictedBoltzmannMachines: RBM, Binary, Spin, Potts,
    mean_from_inputs, initialize!, onehot_decode, onehot_encode

@testset "initialization Binary" begin
    data = rand(2, 3, 10^6) .≤ 3/4
    rbm = RBM(Binary((2,3)), Binary((0,)), randn(2,3,0))
    initialize!(rbm, data)
    @test mean_from_inputs(rbm.visible) ≈ reshape(mean(data; dims=3), size(rbm.visible)) rtol=0.01
end

@testset "initialization Spin" begin
    data = Int8.(sign.(rand(2, 3, 10^6) .- 1/4))
    rbm = RBM(Spin((2,3)), Binary((0,)), randn(2,3,0))
    initialize!(rbm, data)
    @test mean_from_inputs(rbm.visible) ≈ reshape(mean(data; dims=3), size(rbm.visible)) rtol=0.01
end

@testset "initialization Potts" begin
    rbm = RBM(Potts((3,2,3)), Binary((0,)), randn(3,2,3,0))
    data = onehot_encode(onehot_decode(rand(3,2,3,10^6) .- rand(3,2,3,1)), 1:3)
    @assert all(sum(data; dims=1) .== 1)
    initialize!(rbm, data)
    @test mean_from_inputs(rbm.visible) ≈ reshape(mean(data; dims=4), size(rbm.visible)) rtol=0.01
end
