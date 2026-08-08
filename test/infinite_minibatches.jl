using Test: @testset, @test, @test_throws
using RestrictedBoltzmannMachines: infinite_minibatches

@testset "unweighted minibatches cycle with fixed batch size" begin
    track = Int[]
    for (i, (x, w)) in zip(1:24, infinite_minibatches(1:10, nothing; batchsize = 3, shuffle = false))
        @test length(x) == 3
        @test isnothing(w)
        append!(track, x)
    end
    @test track == repeat(1:9, 8) # the partial batch [10] is dropped each epoch

    for (i, (x, w)) in zip(1:24, infinite_minibatches(1:3, nothing; batchsize = 3, shuffle = false))
        @test x == 1:3
        @test isnothing(w)
    end
end

@testset "weighted minibatches slice data and weights jointly" begin
    data = reshape(1.0:20.0, 2, 10)
    wts = collect(1:10)
    batches = collect(Iterators.take(infinite_minibatches(data, wts; batchsize = 3, shuffle = false), 3))
    @test [w for (x, w) in batches] == [1:3, 4:6, 7:9]
    @test [x for (x, w) in batches] == [data[:, 1:3], data[:, 4:6], data[:, 7:9]]

    # wts[i] == i, so each weight batch reveals the indices used for the data batch
    for (x, w) in Iterators.take(infinite_minibatches(data, wts; batchsize = 2, shuffle = true), 20)
        @test x == data[:, w]
    end
end

@testset "shuffle draws a fresh permutation each epoch" begin
    orders = Set{Vector{Int}}()
    for (x, w) in Iterators.take(infinite_minibatches(1:10, collect(1:10); batchsize = 10, shuffle = true), 20)
        @test sort(x) == 1:10
        @test x == w
        push!(orders, copy(x))
    end
    @test length(orders) > 1
end

@testset "batchsize edge cases" begin
    @test_throws ArgumentError infinite_minibatches(1:3, nothing; batchsize = 0, shuffle = false)
    @test_throws ArgumentError infinite_minibatches(1:3, collect(1:3); batchsize = 0, shuffle = false)

    # batchsize larger than the data yields an empty stream
    @test isnothing(iterate(infinite_minibatches(randn(2, 5), nothing; batchsize = 6, shuffle = false)))
    @test isnothing(iterate(infinite_minibatches(randn(2, 5), rand(5); batchsize = 6, shuffle = false)))
end
