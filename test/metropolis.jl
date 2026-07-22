using Test: @test, @testset, @inferred
using Statistics: mean, std, var, cor
using Random: randn!, bitrand
using LogExpFunctions: softmax
using RestrictedBoltzmannMachines: BinaryRBM, energy, free_energy, metropolis, metropolis!
import RestrictedBoltzmannMachines as RBMs

@testset "metropolis β=$β" for β in [0.5, 1.0, 2.0]
    N = 5
    M = 2
    rbm = BinaryRBM(randn(N), randn(M), randn(N, M) / √N)
    T = 10000
    B = 100
    v = bitrand(N, B, T)
    for t in 2:T
        v[:, :, t] .= metropolis(rbm, v[:, :, t - 1]; β)
    end
    counts = Dict{BitVector, Int}()
    for t in 1000:T, n in 1:B
        counts[v[:, n, t]] = get(counts, v[:, n, t], 0) + 1
    end
    freqs = Dict(v => c / sum(values(counts)) for (v, c) in counts)
    𝒱 = [BitVector(digits(Bool, x; base = 2, pad = N)) for x in 0:(2^N - 1)]
    @test cor([get(freqs, v, 0.0) for v in 𝒱], softmax(-β * free_energy.(Ref(rbm), 𝒱))) > 0.99
end

@testset "metropolis! β=$β" for β in [0.5, 1.0, 2.0]
    N = 5
    M = 2
    rbm = BinaryRBM(randn(N), randn(M), randn(N, M) / √N)
    T = 10000
    B = 100
    v = bitrand(N, B, T)
    metropolis!(v, rbm; β)
    counts = Dict{BitVector, Int}()
    for t in 1000:T, n in 1:B
        counts[v[:, n, t]] = get(counts, v[:, n, t], 0) + 1
    end
    freqs = Dict(v => c / sum(values(counts)) for (v, c) in counts)
    𝒱 = [BitVector(digits(Bool, x; base = 2, pad = N)) for x in 0:(2^N - 1)]
    @test cor([get(freqs, v, 0.0) for v in 𝒱], softmax(-β * free_energy.(Ref(rbm), 𝒱))) > 0.99
end
