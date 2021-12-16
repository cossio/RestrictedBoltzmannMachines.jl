include("tests_init.jl")

@testset "initialization Binary" begin
    data = rand(2, 3, 10^6) .≤ 3/4
    μd = RBMs.mean_(data; dims=3)
    rbm = RBMs.RBM(RBMs.Binary(2,3), RBMs.Binary(0), randn(2,3,0))
    RBMs.initialize!(rbm, data)
    μm = 1 ./ (1 .+ exp.(-rbm.visible.θ))
    @test μm ≈ μd rtol=0.01
end

@testset "initialization Spin" begin
    data = sign.(rand(2, 3, 10^6) .- 1/4)
    μd = RBMs.mean_(data; dims=3)
    rbm = RBMs.RBM(RBMs.Spin(2,3), RBMs.Binary(0), randn(2,3,0))
    RBMs.initialize!(rbm, data)
    μm = tanh.(rbm.visible.θ)
    @test μm ≈ μd rtol=0.01
end

@testset "initialization Potts" begin
    data = RBMs.onehot_encode(RBMs.onehot_decode(rand(3,2,3,10^6) .- rand(3,2,3,1)), 1:3)
    @assert all(sum(data; dims=1) .== 1)
    μd = RBMs.mean_(data; dims=4)
    rbm = RBMs.RBM(RBMs.Potts(3,2,3), RBMs.Binary(0), randn(3,2,3,0))
    RBMs.initialize!(rbm, data)
    μm = LogExpFunctions.softmax(rbm.visible.θ; dims=1)
    @test μm ≈ μd rtol=0.01
end
