include("tests_init.jl")

@testset "obs" begin
    X, Y = randn(10, 4), randn(5, 4)
    @test (@inferred RBMs._nobs(X, Y)) == 4
    @test_throws AssertionError RBMs._nobs(X, randn(5, 3))
    @test (@inferred RBMs._getobs([2, 3], X, Y)) == (X[:, [2, 3]], Y[:, [2, 3]])
end

@testset "minibatches" begin
    @test RBMs.minibatch_count(10; batchsize = 3) == 4
    @test RBMs.minibatches(10; batchsize=3) == [[1,2,3], [4,5,6], [7,8,9], [10,1,2]]
    @test RBMs.minibatches(10; batchsize=5) == [[1,2,3,4,5], [6,7,8,9,10]]
    @test length(RBMs.minibatches(10; batchsize=3)) == RBMs.minibatch_count(10; batchsize=3)
end
