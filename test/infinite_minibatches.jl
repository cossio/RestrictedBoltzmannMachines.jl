using Test: @testset, @test, @test_throws, @inferred
using EllipsisNotation: (..)
using FillArrays: Trues
using RestrictedBoltzmannMachines: nobs, getobs, shuffleobs, infinite_minibatches

@testset "nobs" begin
    @test @inferred(nobs(randn(3, 4, 5))) == 5
    @test @inferred(nobs(randn(3, 4, 5), randn(2, 3, 5))) == 5
    @test @inferred(nobs(randn(3, 4, 5), Trues(5))) == 5
end

@testset "getobs" begin
    X = randn(3, 4, 7, 5)
    Y = randn(2, 3, 5)
    @test @inferred(getobs(2, X, Y)) == (X[.., 2], Y[.., 2])
    @test @inferred(getobs(1:2, X, Y)) == (X[.., 1:2], Y[.., 1:2])

    # lazy uniform weights stay lazy under minibatch slicing
    w = Trues(5)
    _, wslice = @inferred getobs(1:2, X, w)
    @test wslice isa Ones
    @test wslice == Trues(2)
end

@testset "shuffleobs" begin
    X, Y = shuffleobs(collect(1:10), collect(1:10))
    @test X == Y
    @test sort(X) == sort(Y) == collect(1:10)

    # lazy uniform weights stay lazy under shuffling
    X, w = shuffleobs(collect(1:10), Trues(10))
    @test sort(X) == collect(1:10)
    @test w isa Ones
end

@testset "infinite_minibatches" begin
    data = 1:10
    track = Int[]
    for (i, (x,)) in zip(1:24, infinite_minibatches(data; batchsize = 3, shuffle = false))
        @test length(x) == 3
        append!(track, x)
    end
    @test track == repeat(1:9, 8)

    data = 1:3
    for (i, (x,)) in zip(1:24, infinite_minibatches(data; batchsize = 3, shuffle = false))
        @test x == 1:3
    end

    @test_throws ArgumentError infinite_minibatches(data; batchsize = 0, shuffle = false)
end

@testset "infinite_minibatches over (data, wts)" begin
    data = randn(2, 10)
    wts = Trues(10)
    for (i, (x, w)) in zip(1:7, infinite_minibatches(data, wts; batchsize = 4, shuffle = false))
        @test size(x) == (2, 4)
        @test w isa Ones
        @test length(w) == 4
    end

    wts = collect(1.0:10.0)
    for (i, (x, w)) in zip(1:7, infinite_minibatches(data, wts; batchsize = 4, shuffle = false))
        @test size(x) == (2, 4)
        @test w == wts[(1:4) .+ 4 * ((i - 1) % 2)]
    end
end

@testset "batchsize larger than the data" begin
    # the training entry point clamps the batchsize before building the
    # iterator (tested through `pcd!` in zero_weight_training.jl)
    data = randn(2, 5)
    @test iterate(infinite_minibatches(data; batchsize = 6, shuffle = false)) === nothing
end
