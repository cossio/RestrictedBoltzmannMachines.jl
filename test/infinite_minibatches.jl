using Test: @testset, @test, @inferred
using EllipsisNotation: (..)
using RestrictedBoltzmannMachines: nobs, getobs, shuffleobs, infinite_minibatches

@testset "nobs" begin
    @test isnothing(@inferred nobs(nothing))
    @test @inferred(nobs(randn(3,4,5))) == 5
    @test @inferred(nobs(nothing, randn(3,4,5))) == 5
    @test @inferred(nobs(randn(3,4,5), nothing)) == 5
    @test @inferred(nobs(randn(3,4,5), randn(2,3,5))) == 5
    @test isnothing(@inferred nobs(nothing, nothing))
    @test isnothing(@inferred nobs())
end

@testset "getobs" begin
    X = randn(3,4,7,5)
    Y = randn(2,3,5)
    @test @inferred(getobs(2, X, Y)) == (X[..,2], Y[..,2])
    @test @inferred(getobs(1:2, X, Y)) == (X[..,1:2], Y[..,1:2])
    @test @inferred(getobs(2, nothing, Y)) == (nothing, Y[..,2])
    @test @inferred(getobs(2, X, nothing)) == (X[..,2], nothing)
    @test @inferred(getobs(1:2, nothing, Y)) == (nothing, Y[..,1:2])
    @test @inferred(getobs(1:2, nothing, nothing)) == (nothing, nothing)
end

@testset "shuffleobs" begin
    X, Y = shuffleobs(1:10, 1:10)
    @test X == Y
    @test sort(X) == sort(Y) == 1:10

    X, Y = shuffleobs(1:10, nothing)
    @test sort(X) == 1:10
    @test isnothing(Y)
end

@testset "infinite_minibatches" begin
    data = 1:10
    track = Int[]
    for (i, (x,)) in zip(1:24, infinite_minibatches(data; batchsize=3, shuffle=false))
        @test length(x) == 3
        append!(track, x)
    end
    @test track == repeat(1:9, 8)
end
