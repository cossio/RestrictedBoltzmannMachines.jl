using Test, Random, LinearAlgebra, Statistics
using NNlib, StatsFuns, Zygote, FiniteDifferences
using RestrictedBoltzmannMachines, OneHot
using RestrictedBoltzmannMachines: meandrop, __transfer_logpdf,  __transfer_pdf

include("../test_utils.jl")

Random.seed!(788)

@testset "Spin energy gradients" begin
    # with batch dimensions
    θ = randn(4,5,6)
    x = rand((-1,1), 4,5,6, 3,2)
    testfun(θ::AbstractArray) = sum(energy(Spin(θ), x))
    (dθ,) = gradient(testfun, θ)
    p = randn(size(θ))
    @test central_fdm(5,1)(ϵ -> testfun(θ + ϵ * p), 0) ≈ sum(dθ .* p)

    # without batch dimensions
    θ = randn(4,5,6)
    x = rand((-1,1), 4,5,6)
    testfun(θ::AbstractArray) = sum(energy(Spin(θ), x))
    (dθ,) = gradient(testfun, θ)
    p = randn(size(θ))
    @test central_fdm(5,1)(ϵ -> testfun(θ + ϵ * p), 0) ≈ sum(dθ .* p)
end

@testset "Spin pdf" begin
    layer = Spin(randn(1))
    Em, Ep = energy(layer, [-1]), energy(layer, [+1])
    @test transfer_pdf(layer, [-1]) ≈ exp(-Em) / (exp(-Em) + exp(-Ep))
    @test transfer_pdf(layer, [+1]) ≈ exp(-Ep) / (exp(-Em) + exp(-Ep))
end

@testset "Spin energy & cgf gradients" begin
    Random.seed!(1)
    θ = randn(4,5,6)
    # with batch dimensions
    x = sign.(randn(4,5,6, 3,2))
    (dI,) = gradient(I -> sum(cgf(Spin(θ), I)), x)
    @test dI ≈ transfer_mean(Spin(θ), x)
    gradtest(θ -> energy(Spin(θ), x), θ)
    gradtest(θ -> cgf(Spin(θ), x), θ)
    # without batch dimensions
    x = randn(4,5,6)
    (dI,) = gradient(I -> sum(cgf(Spin(θ), I)), x)
    @test dI ≈ transfer_mean(Spin(θ), x)
    gradtest(θ -> energy(Spin(θ), x), θ)
    gradtest(θ -> cgf(Spin(θ), x), θ)
end

layer = Spin(randn(10,5))
sample = random(layer, zeros(size(layer)..., 10000))
@test transfer_mean(layer) ≈ mean(sample; dims=3) rtol=0.1
@test transfer_std(layer) ≈ std(sample; dims=3) rtol=0.1
@test transfer_var(layer) ≈ var(sample; dims=3) rtol=0.1
@test transfer_mean_abs(layer) ≈ mean(abs, sample; dims=3) rtol=0.1
