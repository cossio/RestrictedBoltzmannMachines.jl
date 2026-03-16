using Test: @test, @testset, @inferred
using Statistics: mean, std, var, cor
using Random: randn!, bitrand
using LogExpFunctions: softmax
using RestrictedBoltzmannMachines: BinaryRBM, energy, free_energy, metropolis, metropolis!, cold_metropolis
import RestrictedBoltzmannMachines as RBMs

@testset "metropolis β=$β" for β in [0.5, 1.0, 2.0]
    N = 5
    M = 2
    rbm = BinaryRBM(randn(N), randn(M), randn(N, M) / √N)
    T = 10000
    B = 100
    v = bitrand(N, B, T)
    for t in 2:T
        v[:,:,t] .= metropolis(rbm, v[:,:,t - 1]; β)
    end
    counts = Dict{BitVector, Int}()
    for t in 1000:T, n in 1:B
        counts[v[:,n,t]] = get(counts, v[:,n,t], 0) + 1
    end
    freqs = Dict(v => c / sum(values(counts)) for (v,c) in counts)
    𝒱 = [BitVector(digits(Bool, x; base=2, pad=N)) for x in 0:(2^N - 1)]
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
        counts[v[:,n,t]] = get(counts, v[:,n,t], 0) + 1
    end
    freqs = Dict(v => c / sum(values(counts)) for (v,c) in counts)
    𝒱 = [BitVector(digits(Bool, x; base=2, pad=N)) for x in 0:(2^N - 1)]
    @test cor([get(freqs, v, 0.0) for v in 𝒱], softmax(-β * free_energy.(Ref(rbm), 𝒱))) > 0.99
end

@testset "cold_metropolis converges to fixed point" begin
    N = 5
    M = 3
    rbm = BinaryRBM(randn(N), randn(M), randn(N, M) / √N)
    v = bitrand(N)
    # Run many cold_metropolis steps to converge
    v1 = cold_metropolis(rbm, v; steps=100)
    # One more step should not change the result (fixed point)
    v2 = cold_metropolis(rbm, v1; steps=1)
    @test v1 == v2
    # Also test with a batch
    vb = bitrand(N, 10)
    vb1 = cold_metropolis(rbm, vb; steps=100)
    vb2 = cold_metropolis(rbm, vb1; steps=1)
    @test vb1 == vb2
end
