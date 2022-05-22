using Test: @test, @testset, @inferred, @test_throws
import RestrictedBoltzmannMachines as RBMs

@testset "obs" begin
    X, Y = randn(10, 4), randn(5, 4)
    @test (@inferred RBMs._nobs(X, Y)) == 4
    @test_throws AssertionError RBMs._nobs(X, randn(5, 3))
    @test (@inferred RBMs._getobs([2, 3], X, Y)) == (X[:, [2, 3]], Y[:, [2, 3]])

    X, Y, Z = randn(10, 4), nothing, randn(5, 4)
    @test (@inferred RBMs._nobs(X, Y, Z)) == 4
    @test_throws AssertionError RBMs._nobs(X, randn(5, 3), nothing)
    @test (@inferred RBMs._getobs([2, 3], X, Y, Z)) == (X[:, [2, 3]], nothing, Z[:, [2, 3]])
end

@testset "minibatches" begin
    @test RBMs.minibatch_count(10; batchsize = 3) == 4
    @test RBMs.minibatches(7; batchsize=3, shuffle=false) == [[1,2,3], [4,5,6], [7,1,2]]
    @test RBMs.minibatches(7; batchsize=5, shuffle=false) == [[1,2,3,4,5], [6,7,1,2,3]]
    @test length(RBMs.minibatches(7; batchsize=3)) == RBMs.minibatch_count(7; batchsize=3)

    X, Y, Z = randn(10, 4), nothing, randn(5, 4)
    batches = RBMs.minibatches(X, Y, Z; batchsize=2, shuffle=false)
    @test length(batches) == RBMs.minibatch_count(4; batchsize=2) == 2
    @test length(batches) == RBMs.minibatch_count(X, Y, Z; batchsize=2)
    @test batches[1] == (X[:,1:2], nothing, Z[:,1:2])
    @test batches[2] == (X[:,3:4], nothing, Z[:,3:4])

    X = randn(7, 10)
    batches = RBMs.minibatches(X, X, copy(X); batchsize=3, shuffle=true)
    for (x1, x2, x3) in batches
        @test size(x1) == (7, 3)
        @test x1 == x2 == x3
    end

    batch = only(RBMs.minibatches([1,2,3], [1,2,3]'; batchsize=3))
    p = sortperm(batch[1])
    @test batch[1][p] == batch[2]'[p] == [1,2,3]
end

@testset "training_epochs" begin
    nsamples = 5421
    nupdates = 10403
    batchsize = 5
    nepochs = RBMs.training_epochs(; nsamples, nupdates, batchsize)
    @test RBMs.minibatch_count(nsamples; batchsize) * (nepochs - 1) ≤ nupdates
    @test RBMs.minibatch_count(nsamples; batchsize) * (nepochs + 1) ≥ nupdates
end
