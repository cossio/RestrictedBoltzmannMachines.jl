import Statistics
import RestrictedBoltzmannMachines as RBMs
using Test: @test, @testset, @inferred
using Statistics: mean, var, cov
using LinearAlgebra: dot
using RestrictedBoltzmannMachines: repeat_size, sizedims, moving_average

@testset "two" begin
    @test RBMs.two(1) === RBMs.two(Int) === 2
    @test RBMs.two(Int8(1)) === RBMs.two(Int8) === Int8(2)
    @test RBMs.two(1f0) === RBMs.two(Float32) === 2f0
    @test RBMs.two(1.0) === RBMs.two(Float64) === 2.0
    @test_throws InexactError RBMs.two(Bool)
    @inferred RBMs.two(1)
end

@testset "inf" begin
    @test_throws InexactError RBMs.inf(1)
    @test Inf === @inferred RBMs.inf(1.0)
    @test Inf32 === @inferred RBMs.inf(1f0)
    @inferred RBMs.inf(1.0)
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
    A = randn(5)
    w = rand(5)
    @test dot(A, w) / sum(w) ≈ @inferred RBMs.wmean(A; wts=w)

    A = randn(4,3,5,2)
    @test mean(A) ≈ @inferred RBMs.wmean(A)
    @test mean(A; dims=(2,4)) ≈ @inferred RBMs.wmean(A; dims=(2,4))

    wts = rand(size(A)...)
    @test sum(A .* wts) ./ sum(wts) ≈ @inferred RBMs.wmean(A; wts)
    @test sum(A .* wts) ./ sum(wts) ≈ @inferred RBMs.wmean(A; wts, dims=:)

    wts = rand(3,2)
    @test sum(reshape(wts,1,3,1,2) .* A; dims=(2,4)) ./ sum(wts) ≈ @inferred RBMs.wmean(A; dims=(2,4), wts)

    wts = rand(2)
    @test sum(reshape(wts,1,1,1,2) .* A; dims=4) ./ sum(wts) ≈ @inferred RBMs.wmean(A; dims=4, wts)
end

@testset "wsum" begin
    A = randn(5)
    w = rand(5)
    @test dot(A, w) ≈ @inferred RBMs.wsum(A; wts=w)

    A = randn(4,3,5,2)
    @test sum(A) ≈ @inferred RBMs.wsum(A)
    @test sum(A; dims=(2,4)) ≈ @inferred RBMs.wsum(A; dims=(2,4))

    wts = rand(size(A)...)
    @test sum(A .* wts) ≈ @inferred RBMs.wsum(A; wts)
    @test sum(A .* wts) ≈ @inferred RBMs.wsum(A; wts, dims=:)

    wts = rand(3,2)
    @test sum(reshape(wts,1,3,1,2) .* A; dims=(2,4)) ≈ @inferred RBMs.wsum(A; dims=(2,4), wts)

    wts = rand(2)
    @test sum(reshape(wts,1,1,1,2) .* A; dims=4) ≈ @inferred RBMs.wsum(A; dims=4, wts)
end

@testset "reshape_maybe" begin
    @test RBMs.reshape_maybe(1, ()) == 1
    @test_throws Exception RBMs.reshape_maybe(1, 1)
    @test_throws Exception RBMs.reshape_maybe(1, (1,))

    @test RBMs.reshape_maybe(fill(1), ()) == 1
    @test RBMs.reshape_maybe(fill(1), (1,)) == [1]
    @test RBMs.reshape_maybe(fill(1), (1,1)) == hcat([1])
    @test_throws Exception RBMs.reshape_maybe(fill(1), (1,2))

    @test RBMs.reshape_maybe([1], ()) == 1
    @test RBMs.reshape_maybe([1], (1,)) == [1]
    @test RBMs.reshape_maybe([1], (1,1)) == hcat([1])
    @test_throws Exception RBMs.reshape_maybe([1], (1,2))

    A = randn(2,2)
    @test RBMs.reshape_maybe(A, 4) == reshape(A, 4)
end

@testset "repeat_size" begin
    ns = ((), (0,), (1,), (1,2), (2,1), (2,3))
    rs = ((), (0,), (1,), (1,2), (2,1), (2,3))
    for n in ns, r in rs
        @test repeat_size(n, r...) == size(repeat(trues(n...), r...))
    end
end

@testset "sizedims" begin
    A = randn(7,4,3,2)
    @test @inferred(sizedims(A)) == ()
    @test @inferred(sizedims(A, 2)) == (4,)
    @test @inferred(sizedims(A, :)) == (7,4,3,2)
    @test @inferred(sizedims(A, (2,))) == (4,)
    @test @inferred(sizedims(A, (2,3))) == (4,3)
    @test @inferred(sizedims(A, 2, 3)) == (4,3)
end

@testset "moving_average" begin
    for n in 1:10, m in 1:10
        @test length(@inferred moving_average(randn(n), m)) == n
    end
    x = randn(11)
    @test @inferred(moving_average(x, length(x)))[6] ≈ mean(x)
    @test @inferred(moving_average(x, 3))[5] ≈ mean([x[4], x[5], x[6]])
    @test @inferred(moving_average(x, 3))[1] ≈ mean([x[1], x[2]])
end
