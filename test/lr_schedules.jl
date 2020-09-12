using Test, Random, Flux, RestrictedBoltzmannMachines
using Flux: Optimise, Optimiser, Params

Random.seed!(84)
w = randn(10, 10)
w_ = randn(10, 10)
loss(x) = Flux.Losses.mse(w*x, w_*x)

@testset "SqrtDecay" begin
    Random.seed!(84)
    w = randn(10, 10)
    w_ = randn(10, 10)
    loss(x) = Flux.Losses.mse(w * x, w_ * x)
    opt = Optimiser(SqrtDecay(; decay=5), ADAM(0.001))
    for t = 1:10^5
        θ = Params([w_])
        x = rand(10)
        θ_ = gradient(() -> loss(x), θ)
        Optimise.update!(opt, θ, θ_)
    end
    @test loss(rand(10, 10)) < 0.01
end

@testset "GeometricDecay" begin
    Random.seed!(2)
    w = randn(10, 10)
    w_ = randn(10, 10)
    loss(x) = Flux.Losses.mse(w * x, w_ * x)
    opt = Optimiser(GeometricDecay(; decay=0.9999), ADAM(0.001))
    for t = 1:10^5
        θ = Params([w_])
        x = rand(10)
        θ_ = gradient(() -> loss(x), θ)
        Optimise.update!(opt, θ, θ_)
    end
    @test loss(rand(10, 10)) < 0.01
end
