using Test: @test, @testset
using Statistics: mean
using Random: bitrand, seed!
using Optimisers: Adam
using LogExpFunctions: softmax
using RestrictedBoltzmannMachines: BinaryRBM, free_energy, initialize!, sample_v_from_v, unbiased_sample, unbiased_estimator, ucd!

@testset "unbiased sampling" begin
    seed!(1234)
    rbm = BinaryRBM(randn(3), randn(2), randn(3, 2) / √3)
    𝒱 = [BitVector(digits(Bool, x; base = 2, pad = 3)) for x in 0:(2^3 - 1)]
    probs = softmax(-free_energy.(Ref(rbm), 𝒱))
    exact_mean = sum(p * Float64.(v) for (p, v) in zip(probs, 𝒱))

    estimate = zeros(3)
    for _ in 1:2000
        sample = unbiased_sample(rbm, falses(3); min_steps = 1, max_steps = 32)
        estimate .+= unbiased_estimator(v -> Float64.(v), sample)
    end
    estimate ./= 2000

    @test all(isapprox.(estimate, exact_mean; atol = 0.05))
end

@testset "ucd" begin
    seed!(1234)
    data = falses(2, 1000)
    data[1, 1:2:end] .= true
    data[2, 1:2:end] .= true

    rbm = BinaryRBM(2, 5)
    initialize!(rbm, data)
    ucd!(rbm, data; iters = 3000, batchsize = 64, nchains = 8, min_steps = 1, max_steps = 32, optim = Adam(5e-4))

    v_sample = sample_v_from_v(rbm, bitrand(2, 10000); steps = 50)

    @test 0.4 < mean(v_sample[1, :]) < 0.6
    @test 0.4 < mean(v_sample[2, :]) < 0.6
    @test 0.4 < mean(v_sample[1, :] .* v_sample[2, :]) < 0.6
end
