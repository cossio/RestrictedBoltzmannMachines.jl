include("tests_init.jl")

@testset "initialization Binary" begin
    rbm = RBMs.RBM(RBMs.Binary(2,3), RBMs.Binary(0), randn(2,3,0))
    data = rand(2, 3, 10^6) .≤ 3/4
    RBMs.initialize!(rbm, data)
    μ = 1 ./ (1 .+ exp.(-rbm.visible.θ))
    @test μ ≈ RBMs.mean_(data; dims=3) rtol=0.01
end

@testset "initialization Spin" begin
    rbm = RBMs.RBM(RBMs.Spin(2,3), RBMs.Binary(0), randn(2,3,0))
    data = sign.(rand(2, 3, 10^6) .- 1/4)
    RBMs.initialize!(rbm, data)
    @test tanh.(rbm.visible.θ) ≈ RBMs.mean_(data; dims=3) rtol=0.01
end

@testset "initialization Potts" begin
    rbm = RBMs.RBM(RBMs.Potts(3,2,3), RBMs.Binary(0), randn(3,2,3,0))
    data = RBMs.onehot_encode(RBMs.onehot_decode(rand(3,2,3,10^6) .- [0.1;0.3;0.1;;;;]), 1:3)
    @assert all(sum(data; dims=1) .== 1)
    RBMs.initialize!(rbm, data)
    @test LogExpFunctions.softmax(rbm.visible.θ) ≈ RBMs.mean_(data; dims=4) rtol=0.01
end
