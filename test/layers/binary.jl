using Test, Random, LinearAlgebra, Statistics
using NNlib, StatsFuns, Zygote, FiniteDifferences
using RestrictedBoltzmannMachines, OneHot
using RestrictedBoltzmannMachines: meandrop

Random.seed!(788)

@testset "Binary energy gradients" begin
    # with batch dimensions
    θ = randn(4,5,6)
    x = rand(Bool, 4,5,6, 3,2)
    testfun(θ::AbstractArray) = sum(energy(Binary(θ), x))
    (dθ,) = gradient(testfun, θ)
    p = randn(size(θ))
    @test central_fdm(5,1)(ϵ -> testfun(θ + ϵ * p), 0) ≈ sum(dθ .* p)

    # without batch dimensions
    θ = randn(4,5,6)
    x = rand(Bool, 4,5,6)
    testfun(θ::AbstractArray) = sum(energy(Binary(θ), x))
    (dθ,) = gradient(testfun, θ)
    p = randn(size(θ))
    @test central_fdm(5,1)(ϵ -> testfun(θ + ϵ * p), 0) ≈ sum(dθ .* p)
end

layer = Binary(randn(10,5))
sample = random(layer, zeros(size(layer)..., 10000))
@test transfer_mean(layer) ≈ mean(sample; dims=3) rtol=0.1
@test transfer_std(layer) ≈ std(sample; dims=3) rtol=0.1
@test transfer_var(layer) ≈ var(sample; dims=3) rtol=0.1
@test transfer_mean_abs(layer) ≈ mean(abs, sample; dims=3) rtol=0.1
