using Test: @test, @testset, @inferred
using Statistics: mean, std, var
using Random: randn!, bitrand
using LogExpFunctions: logsumexp
using RestrictedBoltzmannMachines: RBM, BinaryRBM, Binary, Spin, Potts, Gaussian, ReLU, dReLU, xReLU, pReLU
using RestrictedBoltzmannMachines: energy, free_energy, sample_from_inputs, sample_v_from_v
using RestrictedBoltzmannMachines: anneal, anneal_zero, ais, aise, raise, log_partition_zero_weight
using RestrictedBoltzmannMachines: logmeanexp, logvarexp, logstdexp
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

@testset "logmeanexp properties" begin
    X = randn(1000, 1000)
    @test only(logmeanexp(logmeanexp(X; dims=1); dims=2)) ≈ logmeanexp(X)
    @test only(logmeanexp(-logmeanexp(-X; dims=1); dims=2)) ≤ only(-logmeanexp(-logmeanexp(X; dims=1); dims=2))
    x = randn()
    @test logmeanexp([x]) ≈ x
    X = randn(1000, 1000, 1)
    @test logmeanexp(X; dims=3) ≈ X
    @test -logmeanexp(-X) ≤ logmeanexp(X)
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
    x = sample_from_inputs(final)
    @test energy(anneal(init, final; β), x) ≈ (1 - β) * energy(init, x) + β * energy(final, x)

    init = Spin(randn(N))
    final = Spin(randn(N))
    x = sample_from_inputs(final)
    @test energy(anneal(init, final; β), x) ≈ (1 - β) * energy(init, x) + β * energy(final, x)

    init = Potts(randn(N))
    final = Potts(randn(N))
    x = sample_from_inputs(final)
    @test energy(anneal(init, final; β), x) ≈ (1 - β) * energy(init, x) + β * energy(final, x)

    init = Gaussian(randn(N), rand(N))
    final = Gaussian(randn(N), rand(N))
    x = sample_from_inputs(final)
    @test energy(anneal(init, final; β), x) ≈ (1 - β) * energy(init, x) + β * energy(final, x)

    init = ReLU(randn(N), rand(N))
    final = ReLU(randn(N), rand(N))
    x = sample_from_inputs(final)
    @test energy(anneal(init, final; β), x) ≈ (1 - β) * energy(init, x) + β * energy(final, x)

    init = dReLU(randn(N), randn(N), rand(N), rand(N))
    final = dReLU(randn(N), randn(N), init.γp, init.γn)
    x = sample_from_inputs(final)
    @test energy(anneal(init, final; β), x) ≈ (1 - β) * energy(init, x) + β * energy(final, x)

    init = pReLU(randn(N), rand(N), randn(N), rand(N) .- 0.5)
    final = pReLU(randn(N), init.γ, randn(N), init.η)
    x = sample_from_inputs(final)
    @test energy(anneal(init, final; β), x) ≈ (1 - β) * energy(init, x) + β * energy(final, x)

    init = xReLU(randn(N), rand(N), randn(N), randn(N))
    final = xReLU(randn(N), init.γ, randn(N), init.ξ)
    x = sample_from_inputs(final)
    @test energy(anneal(init, final; β), x) ≈ (1 - β) * energy(init, x) + β * energy(final, x)
end

@testset "anneal_zero" begin
    β = 0.3
    N = 11
    M = 5
    rbm = BinaryRBM(randn(N), randn(M), randn(N,M))
    init = Binary(randn(N))
    null = BinaryRBM(init.θ, zeros(M), zeros(N,M))
    v = bitrand(N, 10)
    h = bitrand(M, 10)
    @test energy(anneal_zero(init, rbm), v, h) ≈ energy(anneal(null, rbm; β=0), v, h)
    @test iszero(anneal_zero(init, rbm).w)

    layer = Binary(randn(N))
    @test iszero(anneal_zero(layer).θ)

    layer = Spin(randn(N))
    @test iszero(anneal_zero(layer).θ)

    layer = Potts(randn(N))
    @test iszero(anneal_zero(layer).θ)

    layer = Gaussian(randn(N), rand(N))
    @test iszero(anneal_zero(layer).θ)
    @test anneal_zero(layer).γ == layer.γ

    layer = ReLU(randn(N), rand(N))
    @test iszero(anneal_zero(layer).θ)
    @test anneal_zero(layer).γ == layer.γ

    layer = dReLU(randn(N), randn(N), rand(N), rand(N))
    @test iszero(anneal_zero(layer).θp)
    @test iszero(anneal_zero(layer).θn)
    @test anneal_zero(layer).γp == layer.γp
    @test anneal_zero(layer).γn == layer.γn

    layer = pReLU(randn(N), rand(N), randn(N), rand(N) .- 0.5)
    @test iszero(anneal_zero(layer).θ)
    @test iszero(anneal_zero(layer).Δ)
    @test anneal_zero(layer).γ == layer.γ
    @test anneal_zero(layer).η == layer.η

    layer = xReLU(randn(N), rand(N), randn(N), randn(N))
    @test iszero(anneal_zero(layer).θ)
    @test iszero(anneal_zero(layer).Δ)
    @test anneal_zero(layer).γ == layer.γ
    @test anneal_zero(layer).ξ == layer.ξ
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
end

@testset "ais" begin
    rbm0 = BinaryRBM(randn(2), randn(1), zeros(2,1))
    rbm1 = BinaryRBM(randn(2), randn(1), randn(2,1))
    v0 = sample_v_from_v(rbm0, bitrand(2, 100000); steps=1)
    data = ais(rbm0, rbm1, v0; nbetas=10)
    @test logmeanexp(data) ≈ log_partition(rbm1) - log_partition(rbm0) rtol=0.1
    @test mean(logmeanexp.(Iterators.partition(data, 10))) ≈ log_partition(rbm1) - log_partition(rbm0) rtol=0.1

    v0 = sample_v_from_v(rbm0, bitrand(2); steps=1)
    data = ais(rbm0, rbm1, v0; nbetas=100000)
    @test only(data) ≈ log_partition(rbm1) - log_partition(rbm0) rtol=0.1
end

@testset "aise" begin
    rbm = BinaryRBM(randn(2), randn(1), randn(2,1))
    lZ = aise(rbm; nbetas=3, nsamples=100000)
    @test logmeanexp(lZ) ≈ log_partition(rbm) rtol=0.1
    lZ = aise(rbm; nbetas=10000, nsamples=10)
    @test logmeanexp(lZ) ≈ log_partition(rbm) rtol=0.1
end

@testset "raise (w = 0)" begin
    rbm = BinaryRBM(randn(2), randn(1), zeros(2,1))
    v = sample_v_from_v(rbm, bitrand(2, 50000); steps=1)
    lZ = raise(rbm; v, nbetas=10)
    @test logmeanexp(lZ) ≈ log_partition(rbm) rtol=0.1
    v = sample_v_from_v(rbm, bitrand(2, 10); steps=1)
    lZ = raise(rbm; v, nbetas=10000)
    @test logmeanexp(lZ) ≈ log_partition(rbm) rtol=0.1
end

@testset "raise" begin
    rbm = BinaryRBM(randn(2), randn(1), randn(2,1))
    v = sample_v_from_v(rbm, bitrand(2, 50000); steps=1000) # need to equilibrate
    lZ = raise(rbm; v, nbetas=10)
    @test logmeanexp(lZ) ≈ log_partition(rbm) rtol=0.1
    @test -logmeanexp(-lZ) ≈ log_partition(rbm) rtol=0.1
end

@testset "aise ≤ lZ ≤ raise" begin
    rbm = BinaryRBM(randn(3), randn(2), randn(3,2))
    lZ = log_partition(rbm)
    v = sample_v_from_v(rbm, bitrand(size(rbm.visible)..., 100_000); steps=500)
    lZf = aise(rbm; nbetas=3, nsamples=size(v)[end])
    lZr = raise(rbm; v, nbetas=3)
    @info @test mean(logmeanexp.(Iterators.partition(lZf, 2))) ≤ lZ ≤ mean(-logmeanexp.(Iterators.partition(-lZr, 2)))
    @test -logmeanexp(-lZr) ≤ logmeanexp(lZr)
end

@testset "ais gaussian" begin
    rbm0 = RBM(Gaussian(randn(20), 1 .+ 5rand(20)), Gaussian(randn(10), 1 .+ rand(10)), randn(20, 10)/10)
    rbm1 = RBM(Gaussian(randn(20), 1 .+ 5rand(20)), Gaussian(randn(10), 1 .+ rand(10)), randn(20, 10)/10)
    lZ0 = log_partition(rbm0)
    lZ1 = log_partition(rbm1)
    v0 = sample_v_from_v(rbm0, randn(20, 10); steps=1)
    R = ais(rbm0, rbm1, v0, nbetas=10000)
    @test logmeanexp(R) ≈ lZ1 - lZ0 rtol=0.1
end

@testset "zero hidden units" begin
    rbm = BinaryRBM(randn(2), randn(0), randn(2,0))
    v = sample_v_from_v(rbm, bitrand(2, 100); steps=1)
    lZ = aise(rbm; nbetas=3, nsamples=100)
    @test logmeanexp(lZ) ≈ log_partition(rbm)
    lZ = raise(rbm; v, nbetas=3)
    @test logmeanexp(lZ) ≈ log_partition(rbm)
end
