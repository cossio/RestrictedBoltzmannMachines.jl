import Statistics
import RestrictedBoltzmannMachines as RBMs
using Test: @test, @testset, @inferred, @test_throws
using Statistics: mean, var, cov
using LinearAlgebra: dot
using RestrictedBoltzmannMachines: convert_eltype

@testset "generate_sequences" begin
    @test collect(RBMs.generate_sequences(2, 1:3)) == reshape(
        [
            [1, 1], [2, 1], [3, 1],
            [1, 2], [2, 2], [3, 2],
            [1, 3], [2, 3], [3, 3],
        ],
        3, 3
    )
end

@testset "wmean" begin
    A = randn(5)
    w = rand(5)
    @test dot(A, w) / sum(w) ≈ @inferred RBMs.wmean(A; wts = w)

    A = randn(4, 3, 5, 2)
    @test mean(A) ≈ @inferred RBMs.wmean(A)
    @test mean(A; dims = (2, 4)) ≈ @inferred RBMs.wmean(A; dims = (2, 4))

    # weights are a vector along the last (sample) dimension
    wts = rand(2)
    w = reshape(wts, 1, 1, 1, 2)
    @test sum(A .* w) / (4 * 3 * 5 * sum(wts)) ≈ @inferred RBMs.wmean(A; wts)
    @test sum(A .* w) / (4 * 3 * 5 * sum(wts)) ≈ @inferred RBMs.wmean(A; wts, dims = :)
    @test sum(A .* w; dims = (2, 4)) ./ (3 * sum(wts)) ≈ @inferred RBMs.wmean(A; dims = (2, 4), wts)
    @test sum(A .* w; dims = 4) ./ sum(wts) ≈ @inferred RBMs.wmean(A; dims = 4, wts)

    # non-finite entries propagate, even when their weight is zero
    @test isnan(RBMs.wmean([NaN, 2.0]; wts = [0.0, 1.0]))
end

@testset "reshape_maybe" begin
    @test RBMs.reshape_maybe(1, ()) == 1
    @test_throws Exception RBMs.reshape_maybe(1, 1)
    @test_throws Exception RBMs.reshape_maybe(1, (1,))

    @test RBMs.reshape_maybe(fill(1), ()) == 1
    @test RBMs.reshape_maybe(fill(1), (1,)) == [1]
    @test RBMs.reshape_maybe(fill(1), (1, 1)) == hcat([1])
    @test_throws Exception RBMs.reshape_maybe(fill(1), (1, 2))

    @test RBMs.reshape_maybe([1], ()) == 1
    @test RBMs.reshape_maybe([1], (1,)) == [1]
    @test RBMs.reshape_maybe([1], (1, 1)) == hcat([1])
    @test_throws Exception RBMs.reshape_maybe([1], (1, 2))

    A = randn(2, 2)
    @test RBMs.reshape_maybe(A, 4) == reshape(A, 4)
end

@testset "convert_eltype" begin
    A = ones(Float32, 5, 4)
    B = @inferred convert_eltype(Float32, A)
    @test B == A
    @test eltype(B) == Float32
    B .= 0
    @test iszero(A) && iszero(B)

    A = ones(Float32, 5, 4)
    B = @inferred convert_eltype(Float64, A)
    @test B ≈ A
    @test eltype(B) == Float64
    B .= 0
    @test iszero(B)
    @test !iszero(A)
end
