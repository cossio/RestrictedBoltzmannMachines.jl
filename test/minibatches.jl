include("tests_init.jl")

@testset "minibatches" begin
    @test minibatch_count(10; batchsize = 3) == 4
    @test minibatches(10; batchsize=3) == [[1,2,3], [4,5,6], [7,8,9], [10,1,2]]
    @test minibatches(10; batchsize=5) == [[1,2,3,4,5], [6,7,8,9,10]]
    @test length(minibatches(10; batchsize=3)) == minibatch_count(10; batchsize=3)
end
