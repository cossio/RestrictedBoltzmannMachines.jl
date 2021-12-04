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
    @test RBMs.mean_(A; dims=(1,3)) ≈ RBMs.sum_(A; dims=(1,3)) ./ (2 * 4)
end

@testset "inf" begin
    @test_throws InexactError RBMs.inf(1)
    @test Inf === @inferred RBMs.inf(1.0)
    @test Inf32 === @inferred RBMs.inf(1f0)
end

@testset "weighted_mean" begin
    @test RBMs.weighted_mean([1,2,3,4], [1,1,2,2]) ≈ (1 + 2 + 2 * (3 + 4))/(1 + 1 + 2 + 2)
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
