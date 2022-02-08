using Test: @test, @testset, @inferred
import Statistics
import RestrictedBoltzmannMachines as RBMs

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
    A = randn(4,3,5,2)
    @test RBMs.wmean(A) ≈ Statistics.mean(A)
    @test RBMs.wmean(A; dims=(2,4)) ≈ Statistics.mean(A; dims=(2,4))
    @inferred RBMs.wmean(A)
    @inferred RBMs.wmean(A; dims=(2,4))

    wts = rand(size(A)...)
    @test RBMs.wmean(A; wts) ≈ sum(A .* wts / sum(wts))
    @inferred RBMs.wmean(A; wts)

    wts = rand(3,2)
    @test RBMs.wmean(A; dims=(2,4), wts) ≈ sum(reshape(wts,1,3,1,2) .* A; dims=(2,4))/sum(wts)
    @inferred RBMs.wmean(A; dims=(2,4), wts)
end

@testset "reshape_maybe" begin
    @test RBMs.reshape_maybe(1, ()) == 1
    @test_throws Exception RBMs.reshape_maybe(1, 1)
    @test_throws Exception RBMs.reshape_maybe(1, (1,))

    @test RBMs.reshape_maybe(fill(1), ()) == 1
    @test RBMs.reshape_maybe(fill(1), (1,)) == [1]
    @test RBMs.reshape_maybe(fill(1), (1,1)) == [1;;]
    @test_throws Exception RBMs.reshape_maybe(fill(1), (1,2))

    @test RBMs.reshape_maybe([1], ()) == 1
    @test RBMs.reshape_maybe([1], (1,)) == [1]
    @test RBMs.reshape_maybe([1], (1,1)) == [1;;]
    @test_throws Exception RBMs.reshape_maybe([1], (1,2))

    A = randn(2,2)
    @test RBMs.reshape_maybe(A, 4) == reshape(A, 4)
end
