using Test, Random, LinearAlgebra, Statistics, DelimitedFiles
import Zygote, Flux, Distributions, SpecialFunctions, LogExpFunctions, QuadGK, NPZ
import RestrictedBoltzmannMachines as RBMs

@testset "two" begin
    @test RBMs.two(1) === RBMs.two(Int) === 2
    @test RBMs.two(Int8(1)) === RBMs.two(Int8) === Int8(2)
    @test RBMs.two(1f0) === RBMs.two(Float32) === 2f0
    @test RBMs.two(1.0) === RBMs.two(Float64) === 2.0
    @test_throws InexactError RBMs.two(Bool)
end

@testset "sum_" begin
    A = randn(2,3,4)
    @test size(RBMs.sum_(A; dims=(1,2))) == (4,)
    @test RBMs.sum_(A; dims=(1,3)) == vec(sum(A; dims=(1,3)))
end

@testset "mean_" begin
    A = randn(2,3,4)
    @inferred RBMs.mean_(A; dims=(1,3))
    @test RBMs.mean_(A; dims=(1,3)) == vec(mean(A; dims=(1,3)))
end

@testset "var_" begin
    A = randn(2,3,4)
    @inferred RBMs.var_(A; dims=(1,3))
    @test RBMs.var_(A; dims=(1,3)) ≈ vec(var(A; dims=(1,3)))
end

@testset "std_" begin
    A = randn(2,3,4)
    @inferred RBMs.std_(A; dims=(1,3))
    @test RBMs.std_(A; dims=(1,3)) ≈ vec(std(A; dims=(1,3)))
end

@testset "inf" begin
    @test_throws InexactError RBMs.inf(1)
    @test Inf === @inferred RBMs.inf(1.0)
    @test Inf32 === @inferred RBMs.inf(1f0)
end

@testset "generate_sequences" begin
    @test collect(RBMs.generate_sequences(2, 1:3)) == reshape(
           [
               [1, 1], [2, 1], [3, 1],
               [1, 2], [2, 2], [3, 2],
               [1, 3], [2, 3], [3, 3]
           ],
           3, 3
        )
end

@testset "broadlike" begin
    A = randn(1,3)
    B = randn(2,1)
    @test RBMs.broadlike(A, B) ≈ A .+ B .- B
    @inferred RBMs.broadlike(A, B)
    @test RBMs.broadlike(A) == A
    @test RBMs.broadlike(A, 1) == A
end

@testset "wmean" begin
    A = randn(4,3,5,2)
    @test RBMs.wmean(A) ≈ mean(A)
    @test RBMs.wmean(A; dims=(2,4)) ≈ mean(A; dims=(2,4))
    @inferred RBMs.wmean(A)
    @inferred RBMs.wmean(A; dims=(2,4))

    wts = rand(size(A)...)
    @test RBMs.wmean(A; wts) ≈ sum(A .* wts / sum(wts))
    @inferred RBMs.wmean(A; wts)

    wts = rand(3,2)
    @test RBMs.wmean(A; dims=(2,4), wts) ≈ sum(reshape(wts,1,3,1,2) .* A; dims=(2,4))/sum(wts)
    @inferred RBMs.wmean(A; dims=(2,4), wts)
end
