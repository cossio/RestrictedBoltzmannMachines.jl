using Test: @test, @testset
using Statistics: mean
using Random: bitrand
using LogStatFunctions: logmeanexp
using RestrictedBoltzmannMachines: RBM, BinaryRBM, Binary, Spin, Potts, Gaussian, ReLU, dReLU,
    xReLU, pReLU, nsReLU,
    energy, sample_from_inputs, sample_v_from_v,
    anneal, anneal_zero, ais, aise, raise, log_partition_zero_weight,
    log_partition

@testset "log_partition_zero_weight" begin
    rbm0 = BinaryRBM(randn(3), randn(2), zeros(3, 2))
    @test log_partition_zero_weight(rbm0) ≈ log_partition(rbm0)
    rbm1 = RBM(rbm0.visible, rbm0.hidden, randn(3, 2))
    @test log_partition_zero_weight(rbm1) ≈ log_partition_zero_weight(rbm0)
end

@testset "anneal layer" begin
    β = 0.3
    N = 11
    for (init, final) in (
            (Binary(; θ = randn(N)), Binary(; θ = randn(N))),
            (Spin(; θ = randn(N)), Spin(; θ = randn(N))),
            (Potts(; θ = randn(N)), Potts(; θ = randn(N))),
            (Gaussian(; θ = randn(N), γ = rand(N)), Gaussian(; θ = randn(N), γ = rand(N))),
            (ReLU(; θ = randn(N), γ = rand(N)), ReLU(; θ = randn(N), γ = rand(N))),
            (dReLU(; θp = randn(N), θn = randn(N), γp = rand(N), γn = rand(N)), dReLU(; θp = randn(N), θn = randn(N), γp = rand(N), γn = rand(N))),
            (pReLU(; θ = randn(N), γ = rand(N), Δ = randn(N), η = rand(N) .- 0.5), pReLU(; θ = randn(N), γ = rand(N), Δ = randn(N), η = rand(N) .- 0.5)),
            (xReLU(; θ = randn(N), γ = rand(N), Δ = randn(N), ξ = randn(N)), xReLU(; θ = randn(N), γ = rand(N), Δ = randn(N), ξ = randn(N))),
            (nsReLU(; θ = randn(N), Δ = randn(N), ξ = randn(N)), nsReLU(; θ = randn(N), Δ = randn(N), ξ = randn(N))),
        )
        x = sample_from_inputs(final)
        @test energy(anneal(init, final; β), x) ≈ (1 - β) * energy(init, x) + β * energy(final, x)
    end
end

@testset "anneal_zero" begin
    N = 11
    M = 5
    rbm = BinaryRBM(randn(N), randn(M), randn(N, M))
    init = Binary(; θ = randn(N))
    null = BinaryRBM(init.θ, zeros(M), zeros(N, M))
    v = bitrand(N, 10)
    h = bitrand(M, 10)
    @test energy(anneal_zero(init, rbm), v, h) ≈ energy(anneal(null, rbm; β = 0), v, h)
    @test iszero(anneal_zero(init, rbm).w)

    # the field-like parameters are zeroed, the others kept
    for (layer, zeroed) in (
            (Binary(; θ = randn(N)), (:θ,)),
            (Spin(; θ = randn(N)), (:θ,)),
            (Potts(; θ = randn(N)), (:θ,)),
            (Gaussian(; θ = randn(N), γ = rand(N)), (:θ,)),
            (ReLU(; θ = randn(N), γ = rand(N)), (:θ,)),
            (dReLU(; θp = randn(N), θn = randn(N), γp = rand(N), γn = rand(N)), (:θp, :θn)),
            (pReLU(; θ = randn(N), γ = rand(N), Δ = randn(N), η = rand(N) .- 0.5), (:θ, :Δ)),
            (xReLU(; θ = randn(N), γ = rand(N), Δ = randn(N), ξ = randn(N)), (:θ, :Δ)),
            (nsReLU(; θ = randn(N), Δ = randn(N), ξ = randn(N)), (:θ, :Δ)),
        )
        z = anneal_zero(layer)
        @test typeof(z) === typeof(layer)
        for f in propertynames(layer)
            if f in zeroed
                @test iszero(getproperty(z, f))
            else
                @test getproperty(z, f) == getproperty(layer, f)
            end
        end
    end
end

@testset "anneal" begin
    N = 10
    M = 7
    β = 0.3
    rbm0 = BinaryRBM(randn(N), randn(M), randn(N, M))
    rbm1 = BinaryRBM(randn(N), randn(M), randn(N, M))
    rbm = anneal(rbm0, rbm1; β)
    v = bitrand(N)
    h = bitrand(M)
    @test energy(rbm, v, h) ≈ (1 - β) * energy(rbm0, v, h) + β * energy(rbm1, v, h)
end

@testset "ais" begin
    rbm0 = BinaryRBM(randn(2), randn(1), zeros(2, 1))
    rbm1 = BinaryRBM(randn(2), randn(1), randn(2, 1))
    v0 = sample_v_from_v(rbm0, bitrand(2, 100000); steps = 1)
    data = ais(rbm0, rbm1, v0; nbetas = 10)
    @test logmeanexp(data) ≈ log_partition(rbm1) - log_partition(rbm0) rtol = 0.1
    @test mean(logmeanexp.(Iterators.partition(data, 10))) ≈ log_partition(rbm1) - log_partition(rbm0) rtol = 0.1

    v0 = sample_v_from_v(rbm0, bitrand(2); steps = 1)
    data = ais(rbm0, rbm1, v0; nbetas = 100000)
    @test only(data) ≈ log_partition(rbm1) - log_partition(rbm0) rtol = 0.1
end

@testset "aise" begin
    rbm = BinaryRBM(randn(2), randn(1), randn(2, 1))
    lZ = aise(rbm; nbetas = 3, nsamples = 100000)
    @test logmeanexp(lZ) ≈ log_partition(rbm) rtol = 0.1
    lZ = aise(rbm; nbetas = 10000, nsamples = 10)
    @test logmeanexp(lZ) ≈ log_partition(rbm) rtol = 0.1
end

@testset "raise (w = 0)" begin
    rbm = BinaryRBM(randn(2), randn(1), zeros(2, 1))
    v = sample_v_from_v(rbm, bitrand(2, 50000); steps = 1)
    lZ = raise(rbm; v, nbetas = 10)
    @test logmeanexp(lZ) ≈ log_partition(rbm) rtol = 0.1
    v = sample_v_from_v(rbm, bitrand(2, 10); steps = 1)
    lZ = raise(rbm; v, nbetas = 10000)
    @test logmeanexp(lZ) ≈ log_partition(rbm) rtol = 0.1
end

@testset "raise" begin
    rbm = BinaryRBM(randn(2), randn(1), randn(2, 1))
    v = sample_v_from_v(rbm, bitrand(2, 50000); steps = 1000) # need to equilibrate
    lZ = raise(rbm; v, nbetas = 10)
    @test logmeanexp(lZ) ≈ log_partition(rbm) rtol = 0.1
    @test -logmeanexp(-lZ) ≈ log_partition(rbm) rtol = 0.1
end

@testset "aise ≤ lZ ≤ raise" begin
    rbm = BinaryRBM(randn(3), randn(2), randn(3, 2))
    lZ = log_partition(rbm)
    v = sample_v_from_v(rbm, bitrand(size(rbm.visible)..., 100_000); steps = 500)
    lZf = aise(rbm; nbetas = 3, nsamples = size(v)[end])
    lZr = raise(rbm; v, nbetas = 3)
    @info @test mean(logmeanexp.(Iterators.partition(lZf, 2))) ≤ lZ ≤ mean(-logmeanexp.(Iterators.partition(-lZr, 2)))
    @test -logmeanexp(-lZr) ≤ logmeanexp(lZr)
end

@testset "ais gaussian" begin
    rbm0 = RBM(
        Gaussian(; θ = randn(20), γ = 1 .+ 5rand(20)),
        Gaussian(; θ = randn(10), γ = 1 .+ 1rand(10)),
        randn(20, 10) / 10
    )
    rbm1 = RBM(
        Gaussian(; θ = randn(20), γ = 1 .+ 5rand(20)),
        Gaussian(; θ = randn(10), γ = 1 .+ 1rand(10)),
        randn(20, 10) / 10
    )
    lZ0 = log_partition(rbm0)
    lZ1 = log_partition(rbm1)
    v0 = sample_v_from_v(rbm0, randn(20, 10); steps = 1)
    R = ais(rbm0, rbm1, v0, nbetas = 10000)
    @test logmeanexp(R) ≈ lZ1 - lZ0 rtol = 0.1
end

@testset "zero hidden units" begin
    rbm = BinaryRBM(randn(2), randn(0), randn(2, 0))
    v = sample_v_from_v(rbm, bitrand(2, 100); steps = 1)
    lZ = aise(rbm; nbetas = 3, nsamples = 100)
    @test logmeanexp(lZ) ≈ log_partition(rbm)
    lZ = raise(rbm; v, nbetas = 3)
    @test logmeanexp(lZ) ≈ log_partition(rbm)
end
