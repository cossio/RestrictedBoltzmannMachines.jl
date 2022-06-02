using Test: @test, @testset, @inferred
using Statistics: mean, std, var
using Random: randn!, bitrand
using LogExpFunctions: logsumexp
using RestrictedBoltzmannMachines: BinaryRBM, energy, free_energy, sample_from_inputs, sample_v_from_v
using RestrictedBoltzmannMachines: anneal, ais, aise, raise, log_partition_zero_weight
using RestrictedBoltzmannMachines: logmeanexp, logvarexp, logstdexp
using RestrictedBoltzmannMachines: RBM, Binary, Spin, Potts, Gaussian, ReLU, dReLU, xReLU, pReLU
using RestrictedBoltzmannMachines: log_partition

@testset "logmeanexp, logvarexp" begin
    A = randn(5,3,2)
    for dims in (2, (1,2), :)
        @test logmeanexp(A; dims) ≈ log.(mean(exp.(A); dims))
        for corrected in (true, false)
            @test logvarexp(A; dims, corrected) ≈ log.(var(exp.(A); dims, corrected))
            @test logstdexp(A; dims, corrected) ≈ log.(std(exp.(A); dims, corrected))
        end
    end
    @test logvarexp(A; dims=2) ≈ log.(var(exp.(A); dims=2))
    @test logstdexp(A; dims=2) ≈ log.(std(exp.(A); dims=2))
    @test logmeanexp(A) ≈ log.(mean(exp.(A)))
    @test logvarexp(A) ≈ log.(var(exp.(A)))
    @test logstdexp(A) ≈ log.(std(exp.(A)))
end

@testset "log_partition_zero_weight" begin
    rbm0 = BinaryRBM(randn(3), randn(2), zeros(3,2))
    @test log_partition_zero_weight(rbm0) ≈ log_partition(rbm0)
    rbm1 = RBM(rbm0.visible, rbm0.hidden, randn(3,2))
    @test log_partition_zero_weight(rbm1) ≈ log_partition_zero_weight(rbm0)
end

@testset "anneal layer" begin
    β = 0.3
    N = 11

    init = Binary(randn(N))
    final = Binary(randn(N))
    null = Binary(zeros(N))
    x = sample_from_inputs(final)
    @test energy(anneal(init, final; β), x) ≈ (1 - β) * energy(init, x) + β * energy(final, x)
    @test energy(anneal(final; β), x) ≈ energy(anneal(null, final; β), x)

    init = Spin(randn(N))
    final = Spin(randn(N))
    null = Spin(zeros(N))
    x = sample_from_inputs(final)
    @test energy(anneal(init, final; β), x) ≈ (1 - β) * energy(init, x) + β * energy(final, x)
    @test energy(anneal(final; β), x) ≈ energy(anneal(null, final; β), x)

    init = Potts(randn(N))
    final = Potts(randn(N))
    null = Potts(zeros(N))
    x = sample_from_inputs(final)
    @test energy(anneal(init, final; β), x) ≈ (1 - β) * energy(init, x) + β * energy(final, x)
    @test energy(anneal(final; β), x) ≈ energy(anneal(null, final; β), x)

    init = Gaussian(randn(N), rand(N))
    final = Gaussian(randn(N), init.γ)
    null = Gaussian(zeros(N), init.γ)
    x = sample_from_inputs(final)
    @test energy(anneal(init, final; β), x) ≈ (1 - β) * energy(init, x) + β * energy(final, x)
    @test energy(anneal(final; β), x) ≈ energy(anneal(null, final; β), x)

    init = ReLU(randn(N), rand(N))
    final = ReLU(randn(N), init.γ)
    null = ReLU(zeros(N), init.γ)
    x = sample_from_inputs(final)
    @test energy(anneal(init, final; β), x) ≈ (1 - β) * energy(init, x) + β * energy(final, x)
    @test energy(anneal(final; β), x) ≈ energy(anneal(null, final; β), x)

    init = dReLU(randn(N), randn(N), rand(N), rand(N))
    final = dReLU(randn(N), randn(N), init.γp, init.γn)
    null = dReLU(zeros(N), zeros(N), init.γp, init.γn)
    x = sample_from_inputs(final)
    @test energy(anneal(init, final; β), x) ≈ (1 - β) * energy(init, x) + β * energy(final, x)
    @test energy(anneal(final; β), x) ≈ energy(anneal(null, final; β), x)

    init = pReLU(randn(N), rand(N), randn(N), rand(N) .- 0.5)
    final = pReLU(randn(N), init.γ, randn(N), init.η)
    null = pReLU(zeros(N), init.γ, zeros(N), init.η)
    x = sample_from_inputs(final)
    @test energy(anneal(init, final; β), x) ≈ (1 - β) * energy(init, x) + β * energy(final, x)
    @test energy(anneal(final; β), x) ≈ energy(anneal(null, final; β), x)

    init = xReLU(randn(N), rand(N), randn(N), randn(N))
    final = xReLU(randn(N), init.γ, randn(N), init.ξ)
    null = xReLU(zeros(N), init.γ, zeros(N), init.ξ)
    x = sample_from_inputs(final)
    @test energy(anneal(init, final; β), x) ≈ (1 - β) * energy(init, x) + β * energy(final, x)
    @test energy(anneal(final; β), x) ≈ energy(anneal(null, final; β), x)
end

@testset "anneal" begin
    N = 10
    M = 7
    β = 0.3
    rbm0 = BinaryRBM(randn(N), randn(M), randn(N,M))
    rbm1 = BinaryRBM(randn(N), randn(M), randn(N,M))
    rbm = anneal(rbm0, rbm1; β)
    v = bitrand(N)
    h = bitrand(M)
    @test energy(rbm, v, h) ≈ (1 - β) * energy(rbm0, v, h) + β * energy(rbm1, v, h)

    rbm0 = BinaryRBM(randn(N), zeros(M), zeros(N,M))
    @test energy(anneal(rbm0.visible, rbm1; β), v, h) ≈ energy(anneal(rbm0, rbm1; β), v, h)
end

@testset "ais" begin
    rbm0 = BinaryRBM(randn(2), randn(1), zeros(2,1))
    rbm1 = BinaryRBM(randn(2), randn(1), randn(2,1))
    v0 = sample_v_from_v(rbm0, bitrand(2, 10000); steps=1)
    data = ais(rbm0, rbm1, v0; nbetas=5)
    @test mean(exp.(data)) ≈ exp(log_partition(rbm1) - log_partition(rbm0)) rtol=0.1
end

@testset "aise" begin
    rbm = BinaryRBM(randn(2), randn(1), randn(2,1))
    lZ = aise(rbm; nbetas=5, nsamples=10000)
    @test mean(exp.(lZ)) ≈ exp(log_partition(rbm)) rtol=0.1
end

@testset "raise (trivial)" begin
    rbm = BinaryRBM(randn(2), randn(1), zeros(2,1))
    v = sample_v_from_v(rbm, bitrand(2, 10000); steps=1)
    lZ = raise(rbm; v, nbetas=5)
    @test mean(exp.(lZ)) ≈ exp(log_partition(rbm)) rtol=0.1
end

@testset "raise" begin
    rbm = BinaryRBM(randn(2), randn(1), randn(2,1))
    v = sample_v_from_v(rbm, bitrand(2, 10000); steps=1000)
    lZ = raise(rbm; v, nbetas=5)
    @test mean(exp.(lZ)) ≈ exp(log_partition(rbm)) rtol=0.1
end
