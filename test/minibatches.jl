include("tests_init.jl")

@testset "minibatches" begin
    @test minibatches(10; batchsize = 3, full = true)  == [1:3, 4:6, 7:9]
    @test minibatches(10; batchsize = 3, full = false) == [1:3, 4:6, 7:9, 10:10]
    @test minibatches(10; batchsize = 3) == minibatches(10; batchsize = 3, full = false)
    @test minibatch_count(10; batchsize = 3, full = true) == 3
    @test minibatch_count(10; batchsize = 3, full = false) == 4
end
