import Random
import RestrictedBoltzmannMachines as RBMs

using Test: @test, @testset, @inferred
using Statistics: mean, std
using LinearAlgebra, Statistics
using LogExpFunctions: softmax

@testset "onehot" begin
    q = 10
    A = rand(1:q, 10, 7, 3)
    X = @inferred RBMs.onehot_encode(A, 1:q)
    @test size(X) == (q, size(A)...)
    for i in CartesianIndices(A), a in 1:q
        @test X[a,i] == (A[i] == a)
    end
    @test A == @inferred RBMs.onehot_decode(X)
end

@testset "categorical_rand" begin
    Random.seed!(1)
    ps = [0.2, 0.5, 0.3]
    counts = Dict{Int,Int}()
    for _ = 1:1000000
        s = RBMs.categorical_rand(ps)
        counts[s] = get(counts, s, 0) + 1
    end
    for (s, c) in counts
        @test c ./ sum(values(counts)) ≈ ps[s] atol=1e-2
    end
end

@testset "categorical_sample" begin
    Random.seed!(2)
    q = 10
    P = rand(q, 4, 3, 7)
    P ./= sum(P; dims=1)
    X = mean(RBMs.onehot_encode(RBMs.categorical_sample(P), 1:q) for _ in 1:10^6)
    @test X ≈ P rtol=0.1
end

@testset "gumbel" begin
    Random.seed!(3)
    @test mean(RBMs.randgumbel() for _ = 1:10^6) ≈ MathConstants.γ rtol=0.01
    @test std(RBMs.randgumbel() for _ = 1:10^6) ≈ π / √6 rtol=0.01
end

@testset "categorical_sample_from_logits" begin
    Random.seed!(52)
    q = 3
    logits = randn(q,2)
    p = mean(RBMs.onehot_encode(RBMs.categorical_sample_from_logits(logits), 1:q) for _ in 1:10^6)
    @test p ≈ softmax(logits; dims=1) rtol=0.01
end

@testset "categorical_sample_from_logits_gumbel" begin
    Random.seed!(52)
    q = 3
    logits = randn(q,2)
    p = mean(RBMs.onehot_encode(RBMs.categorical_sample_from_logits_gumbel(logits), 1:q) for _ in 1:10^6)
    @test p ≈ softmax(logits; dims=1) rtol=0.01
end
