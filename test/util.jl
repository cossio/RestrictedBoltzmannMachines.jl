include("tests_init.jl")

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

@testset "batch_mean" begin
    @test RBMs.batch_mean([1,2,3,4], [1,1,2,2]) ≈ (1 + 2 + 2 * (3 + 4))/(1 + 1 + 2 + 2)
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

@testset "maybe_scalar" begin
    RBMs.maybe_scalar(1) == 1
    RBMs.maybe_scalar(fill(1)) == 1
    RBMs.maybe_scalar(I(2)) == I(2)
    RBMs.maybe_scalar([1; 2]) == [1; 2]
end
