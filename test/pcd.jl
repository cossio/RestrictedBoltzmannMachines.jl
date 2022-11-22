using Test: @test, @testset
using Statistics: mean
using Random: bitrand
using Optimisers: Adam
using RestrictedBoltzmannMachines: RBM, BinaryRBM, sample_v_from_v, initialize!, pcd!

@testset "pcd" begin
    data = falses(2, 1000)
    data[1, 1:2:end] .= true
    data[2, 1:2:end] .= true

    rbm = BinaryRBM(2, 5)
    initialize!(rbm, data)
    pcd!(rbm, data; iters = 10000, batchsize = 64, steps = 10, optim = Adam(5e-4))

    v_sample = sample_v_from_v(rbm, bitrand(2, 10000); steps=50)

    @test 0.4 < mean(v_sample[1,:]) < 0.6
    @test 0.4 < mean(v_sample[2,:]) < 0.6
    @test 0.4 < mean(v_sample[1,:] .* v_sample[2,:]) < 0.6
end
