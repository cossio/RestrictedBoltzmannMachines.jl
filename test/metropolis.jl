using Test: @test, @testset, @inferred
using Statistics: mean, std, var, cor
using Random: randn!, bitrand
using LogExpFunctions: logsumexp
using RestrictedBoltzmannMachines: BinaryRBM, energy, free_energy, metropolis!
import RestrictedBoltzmannMachines as RBMs

@testset "metropolis!" begin
    N = 5
    M = 2
    rbm = BinaryRBM(randn(N), randn(M), randn(N, M) / √N)

    β = 2
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
    ℱ = free_energy.(Ref(rbm), 𝒱)

    @test cor([get(freqs, v, 0.0) for v in 𝒱], exp.(-β * ℱ .- logsumexp(-β * ℱ))) > 0.99
end
