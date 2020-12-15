using Test, Random, StatsFuns, LinearAlgebra
using NNlib: softmax
using RestrictedBoltzmannMachines
using RestrictedBoltzmannMachines: sumdrop, sumdropfirst, meandrop, wmean,
    logsumexp, logsumexpdrop,
    tail2, front2, allequal,
    odditems, evenitems, oddargs, evenargs, interleave, erfx, tuplejoin,
    inf, nan, _fieldnames, tuplesub, staticgetindex, tuplefill, Δ2, two,
    unzip, log1msoftmax, seqgen, scalarize

@testset "two" begin
    @test two(1) === two(Int) === 2
    @test two(Int8(1)) === two(Int8) === Int8(2)
    @test two(1f0) === two(Float32) === 2f0
    @test two(1.0) === two(Float64) === 2.0
    @test_throws InexactError two(Bool)
end

@testset "<|" begin
    @test sin <| 1 == sin(1)
end

@testset "sumdrop / meandrop" begin
    A = randn(2,3,4)
    @test size(sumdrop(A; dims=(1,2))) == (4,)
    @test sumdrop(A; dims=(1,3)) == vec(sum(A; dims=(1,3)))
    @test sumdropfirst(A, Val(2)) == vec(sum(A; dims=(1,2)))
end

@testset "meandrop" begin
    A = randn(2,3,4)
    @inferred meandrop(A; dims=(1,3))
    @test meandrop(A; dims=(1,3)) ≈ sumdrop(A; dims=(1,3)) ./ (2 * 4)
end

@testset "logsumexp & softmax" begin
    A = randn(2,3,4)
    @test logsumexp(A; dims=(1,2)) ≈ log.(sum(exp.(A); dims=(1,2)))
    @test logsumexpdrop(A; dims=(1,2)) ≈ log.(sumdrop(exp.(A); dims=(1,2)))
    @test size(logsumexp(A; dims=(1,2))) == size(sum(A; dims=(1,2)))
    @inferred logsumexp(A; dims=(1,2))
    @test size(logsumexpdrop(A; dims=(1,2))) == size(sumdrop(A; dims=(1,2)))
    @inferred logsumexpdrop(A; dims=(1,2))
end

@testset "tuple" begin
    @test tail2((1,2)) == front2((1,2)) == ()
    @test tail2((1,2,3,4,5,6)) == (3,4,5,6)
    @test front2((1,2,3,4,5,6)) == (1,2,3,4)
    @test odditems((1,2,3,4,5,6)) == oddargs(1,2,3,4,5,6) == (1,3,5)
    @test evenitems((1,2,3,4,5,6)) == evenargs(1,2,3,4,5,6) == (2,4,6)
    @test interleave(0, (1,2,3,4)) == (0,1,0,2,0,3,0,4,0)
end

@testset "allequal" begin
    @test allequal()
    @test allequal(1)
    @test !allequal(1,2)
    @test !allequal(1,2,3)
    @test !allequal(1,2,1,1)
    @test allequal(1,1,1)
    @test allequal(1,1,1,1)
end

@testset "erfx" begin
    @inferred erfx(0)
    @inferred erfx(1)
    @inferred erfx(0f0)
    @inferred erfx(0e0)
    @inferred erfx(NaN)
    @test erfx(0) ≈ 2 / √π
    @test erfx(0.0) ≈ 2 / √π
    @test erfx(+1e-2) ≈ 1.12834155558496169159095235481
    @test erfx(-1e-2) ≈ 1.12834155558496169159095235481
    @test isnan(erfx(NaN))
    @test iszero(erfx(+Inf))
    @test iszero(erfx(-Inf))
end

@testset "tuplejoin" begin
    @test tuplejoin((1,2), (2,)) == (1,2,2)
    @test tuplejoin((1,2), (3,2)) == (1,2,3,2)
    @test tuplejoin((1,2), (3,2), ("hola",1)) == (1,2,3,2,"hola",1)
    @test tuplejoin((1,2)) == (1,2)
    @test tuplejoin() == ()
end

@testset "inf & nan" begin
    @test_throws InexactError inf(1)
    @test_throws InexactError nan(1)
    @test inf(1.0) === Inf
    @test inf(1f0) === Inf32
    @test nan(1.0) === NaN
    @test nan(1f0) === NaN32
end

@testset "_fieldnames" begin
    @test _fieldnames(Rational) == (:num, :den)
    @inferred _fieldnames(Rational)
end

@testset "tuplesub" begin
    @test tuplesub((1, "hola", 4, 5, 'a'), Val(2), Val(4)) == ("hola", 4, 5)
    @inferred tuplesub((1, "hola", 4, 5, 'a'), Val(2), Val(4))
end

@testset "tuplefill" begin
    @test tuplefill(1, Val(5)) == (1,1,1,1,1)
    @inferred tuplefill(1, Val(5))
end

@testset "staticgetindex" begin
    @test staticgetindex((1, "hola", 4, 5, 'a'), Val(2:4)) == ("hola", 4, 5)
    @inferred staticgetindex((1, "hola", 4, 5, 'a'), Val(2:4))
end

@testset "Δ2" begin
    @test Δ2(5, 4) ≈ 9
end

@testset "wmean" begin
    @test wmean([1,2,3,4], [1,1,2,2]) ≈ (1 + 2 + 2 * (3 + 4)) / (1 + 1 + 2 + 2)
end

@testset "unzip" begin
    @test unzip([(1,2,3), (5,6,7)]) == ([1, 5], [2, 6], [3, 7])
    @inferred unzip([(1,2,3), (5,6,7)])
end

@testset "throttlen" begin
    arr = Int[]
    f = throttlen(2) do x
        push!(arr, x)
    end
    for i = 1:10
       @inferred f(i)
    end
    @test arr == 2:2:10
end

@testset "log1msoftmax" begin
    x = randn(4,4,2)
    @test log1msoftmax(x; dims=1) ≈ log.(1 .- exp.(x) ./ sum(exp.(x); dims=1))
    @test log1msoftmax(x; dims=(1,2)) ≈ log.(1 .- exp.(x) ./ sum(exp.(x); dims=(1,2)))
end

@testset "seqgen" begin
    @test collect(seqgen(2, 1:3)) == reshape(
           [ [1, 1], [2, 1], [3, 1],
             [1, 2], [2, 2], [3, 2],
             [1, 3], [2, 3], [3, 3]
           ],
         3,3)
end

@testset "scalarize" begin
    A = randn(4,4)
    @test scalarize(A) == A
    @test scalarize(zeros()) == 0.0
end

@testset "dotcos" begin
    x = randn(5)
    y = randn(5)
    @test dotcos(x, y) ≈ dot(x, y) / norm(x) / norm(y)
end
