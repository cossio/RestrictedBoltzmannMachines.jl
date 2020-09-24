using Test, Random, LinearAlgebra, Statistics
using NNlib, StatsFuns, Zygote, FiniteDifferences
using RestrictedBoltzmannMachines, OneHot
using RestrictedBoltzmannMachines: meandrop, __transfer_logpdf,  __transfer_pdf

include("../test_utils.jl")

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

@testset "Binary pdf" begin
    layer = Binary(randn(1))
    E0, E1 = energy(layer, [false]), energy(layer, [true])
    @test transfer_pdf(layer, [false]) ≈ exp(-E0) / (exp(-E0) + exp(-E1))
    @test transfer_pdf(layer, [true])  ≈ exp(-E1) / (exp(-E0) + exp(-E1))
end

@testset "Binary energy & cgf gradients" begin
    Random.seed!(1)
    θ = randn(4,5,6)
    # with batch dimensions
    x = map(x -> x ≤ 0 ? 0 : 1, randn(4,5,6, 3,2))
    (dI,) = gradient(I -> sum(cgf(Binary(θ), I)), x)
    @test dI ≈ transfer_mean(Binary(θ), x)
    gradtest(θ -> energy(Binary(θ), x), θ)
    gradtest(θ -> cgf(Binary(θ), x), θ)
    # without batch dimensions
    x = randn(4,5,6)
    (dI,) = gradient(I -> sum(cgf(Binary(θ), I)), x)
    @test dI ≈ transfer_mean(Binary(θ), x)
    gradtest(θ -> energy(Binary(θ), x), θ)
    gradtest(θ -> cgf(Binary(θ), x), θ)
end

layer = Binary(randn(10,5))
sample = random(layer, zeros(size(layer)..., 10000))
@test transfer_mean(layer) ≈ mean(sample; dims=3) rtol=0.1
@test transfer_std(layer) ≈ std(sample; dims=3) rtol=0.1
@test transfer_var(layer) ≈ var(sample; dims=3) rtol=0.1
@test transfer_mean_abs(layer) ≈ mean(abs, sample; dims=3) rtol=0.1
