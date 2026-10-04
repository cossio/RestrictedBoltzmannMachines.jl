using Test: @test, @testset
using Statistics: cor
using Random: bitrand, seed!
using LogExpFunctions: softmax
using RestrictedBoltzmannMachines: BinaryRBM, free_energy, metropolis, metropolis!, cold_metropolis

# the empirical distribution of the chains after burn-in matches the Boltzmann distribution at β
function check_metropolis_histogram(rbm, v, β)
    counts = Dict{BitVector, Int}()
    for t in 1000:size(v, 3), n in 1:size(v, 2)
        counts[v[:, n, t]] = get(counts, v[:, n, t], 0) + 1
    end
    freqs = Dict(v => c / sum(values(counts)) for (v, c) in counts)
    N = size(v, 1)
    𝒱 = [BitVector(digits(Bool, x; base = 2, pad = N)) for x in 0:(2^N - 1)]
    return @test cor([get(freqs, v, 0.0) for v in 𝒱], softmax(-β * free_energy.(Ref(rbm), 𝒱))) > 0.99
end

@testset "$name β=$β" for β in [0.5, 1.0, 2.0], (name, run!) in (
            (
                "metropolis", (v, rbm, β) -> for t in 2:size(v, 3)
                    v[:, :, t] .= metropolis(rbm, v[:, :, t - 1]; β)
            end,
            ),
            ("metropolis!", (v, rbm, β) -> metropolis!(v, rbm; β)),
        )
    N = 5
    M = 2
    rbm = BinaryRBM(randn(N), randn(M), randn(N, M) / √N)
    v = bitrand(N, 100, 10000)
    run!(v, rbm, β)
    check_metropolis_histogram(rbm, v, β)
end

@testset "metropolis accepts a single unbatched configuration" begin
    N = 5
    M = 3
    rbm = BinaryRBM(randn(N), randn(M), randn(N, M) / √N)
    v = bitrand(N)
    v1 = metropolis(rbm, v; β = 0.5, steps = 3)
    @test v1 isa BitVector
    @test size(v1) == (N,)
    # a chain of unbatched steps samples the same distribution as the batched sampler
    vt = bitrand(N, 1, 20000)
    for t in 2:size(vt, 3)
        vt[:, 1, t] .= metropolis(rbm, vt[:, 1, t - 1]; β = 0.5)
    end
    check_metropolis_histogram(rbm, vt, 0.5)
end

@testset "cold_metropolis converges to fixed point" begin
    #= The zero-temperature dynamics v -> mode_v(mean_h(v)) is deterministic but can in
    principle land on a 2-cycle instead of a fixed point for an unlucky random RBM, so
    seed the RNG to keep this test deterministic. =#
    seed!(87)
    N = 5
    M = 3
    rbm = BinaryRBM(randn(N), randn(M), randn(N, M) / √N)
    v = bitrand(N)
    v1 = cold_metropolis(rbm, v; steps = 100)
    v2 = cold_metropolis(rbm, v1; steps = 1)
    @test v1 == v2
    vb = bitrand(N, 10)
    vb1 = cold_metropolis(rbm, vb; steps = 100)
    vb2 = cold_metropolis(rbm, vb1; steps = 1)
    @test vb1 == vb2
end
