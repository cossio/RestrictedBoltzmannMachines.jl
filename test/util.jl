include("tests_init.jl")

@testset "two" begin
    @test two(1) === two(Int) === 2
    @test two(Int8(1)) === two(Int8) === Int8(2)
    @test two(1f0) === two(Float32) === 2f0
    @test two(1.0) === two(Float64) === 2.0
    @test_throws InexactError two(Bool)
end

@testset "sum_" begin
    A = randn(2,3,4)
    @test size(sum_(A; dims=(1,2))) == (4,)
    @test sum_(A; dims=(1,3)) == vec(sum(A; dims=(1,3)))
end

@testset "mean_" begin
    A = randn(2,3,4)
    @inferred mean_(A; dims=(1,3))
    @test mean_(A; dims=(1,3)) ≈ sum_(A; dims=(1,3)) ./ (2 * 4)
end

@testset "inf" begin
    @test_throws InexactError inf(1)
    @test inf(1.0) === Inf
    @test inf(1f0) === Inf32
end

@testset "weighted_mean" begin
    @test weighted_mean([1,2,3,4], [1,1,2,2]) ≈ (1 + 2 + 2 * (3 + 4)) / (1 + 1 + 2 + 2)
end

@testset "generate_sequences" begin
    @test collect(generate_sequences(2, 1:3)) == reshape(
           [ [1, 1], [2, 1], [3, 1],
             [1, 2], [2, 2], [3, 2],
             [1, 3], [2, 3], [3, 3]
           ],
         3,3)
end
