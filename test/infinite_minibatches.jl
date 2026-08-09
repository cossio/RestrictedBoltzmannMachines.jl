using Test: @testset, @test, @test_logs, @test_throws
using FillArrays: Trues
using RestrictedBoltzmannMachines: infinite_minibatches

@testset "minibatches cycle with fixed batch size" begin
    track = Int[]
    for (i, (x,)) in zip(1:24, infinite_minibatches(1:10; batchsize = 3, shuffle = false))
        @test length(x) == 3
        append!(track, x)
    end
    @test track == repeat(1:9, 8) # the partial batch [10] is dropped each epoch

    for (i, (x,)) in zip(1:24, infinite_minibatches(1:3; batchsize = 3, shuffle = false))
        @test x == 1:3
    end
end

@testset "minibatches slice data and weights jointly" begin
    data = reshape(1.0:20.0, 2, 10)
    wts = collect(1:10)
    batches = collect(Iterators.take(infinite_minibatches(data, wts; batchsize = 3, shuffle = false), 3))
    @test [w for (x, w) in batches] == [1:3, 4:6, 7:9]
    @test [x for (x, w) in batches] == [data[:, 1:3], data[:, 4:6], data[:, 7:9]]

    # wts[i] == i, so each weight batch reveals the indices used for the data batch
    for (x, w) in Iterators.take(infinite_minibatches(data, wts; batchsize = 2, shuffle = true), 20)
        @test x == data[:, w]
    end

    # lazy uniform weights stay lazy under minibatch slicing
    for (x, w) in Iterators.take(infinite_minibatches(data, Trues(10); batchsize = 4, shuffle = false), 7)
        @test size(x) == (2, 4)
        @test w isa Trues
        @test length(w) == 4
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
    @test_throws ArgumentError infinite_minibatches(1:3; batchsize = 0, shuffle = false)
    @test_throws ArgumentError infinite_minibatches(1:3, collect(1:3); batchsize = 0, shuffle = false)

    # batchsize larger than the data clamps to one full batch, with a warning
    # from the loader (the training entry point clamps before building the
    # iterator, so training never hits this warning)
    it = @test_logs (:warn,) infinite_minibatches(randn(2, 5), rand(5); batchsize = 6, shuffle = false)
    (x, w), _ = iterate(it)
    @test size(x) == (2, 5)
    @test length(w) == 5
end
