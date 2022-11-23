import Random
using Test: @test, @testset, @inferred
using Statistics: mean, var
using Random: bitrand, rand!, randn!
using LinearAlgebra: norm
using RestrictedBoltzmannMachines: RBM, Binary, free_energy, Gaussian, ReLU, dReLU, pReLU, xReLU,
    sample_v_from_v, sample_h_from_h, mean_from_inputs, var_from_inputs,
    rescale_hidden!, rescale_activations!, rescale_weights!,  weight_norms

Random.seed!(23)

@testset "rescale_activations!" begin
    N = (7,2)
    layers = (
        Gaussian(; θ = randn(N...), γ = randn(N...)),
        ReLU(; θ = randn(N...), γ = randn(N...)),
        dReLU(; θp = randn(N...), θn = randn(N...), γp = randn(N...), γn = randn(N...)),
        pReLU(; θ = randn(N...), γ = randn(N...), Δ = randn(N...), η = 2rand(N...) .- 1),
        xReLU(; θ = randn(N...), γ = randn(N...), Δ = randn(N...), ξ = randn(N...))
    )
    λ = rand(N...)
    for layer in layers
        μ = mean_from_inputs(layer)
        ν = var_from_inputs(layer)
        rescale_activations!(layer, λ)
        @test mean_from_inputs(layer) ≈ μ ./ λ
        @test var_from_inputs(layer) ≈ ν ./ λ.^2
    end
end

@testset "rescale_hidden!" begin
    rbm = RBM(Binary((2,)), ReLU((1,)), randn(2,1))
    randn!(rbm.visible.θ)
    randn!(rbm.hidden.θ)
    rand!(rbm.hidden.γ)
    rbm.hidden.γ .+= 0.5

    v = sample_v_from_v(rbm, bitrand(size(rbm.visible)..., 20000); steps=100)
    h = sample_h_from_h(rbm, rand(size(rbm.hidden)..., 20000); steps=100)
    ave_v = mean(v; dims=2)
    ave_h = mean(h; dims=2)
    var_v = var(v; dims=2)
    var_h = var(h; dims=2)
    F = free_energy(rbm, v)

    λ = [1 + rand()]
    rescale_hidden!(rbm, λ)

    @test free_energy(rbm, v) ≈ F .+ sum(log, λ)

    v = sample_v_from_v(rbm, bitrand(size(rbm.visible)..., 20000); steps=100)
    h = sample_h_from_h(rbm, rand(size(rbm.hidden)..., 20000); steps=100)

    @test mean(v; dims=2) ≈ ave_v rtol=0.1
    @test mean(h; dims=2) ≈ ave_h ./ λ rtol=0.1
    @test var(v; dims=2) ≈ var_v rtol=0.1
    @test var(h; dims=2) ≈ var_h ./ λ.^2 rtol=0.1
end

@testset "rescale_weights!" begin
    rbm = RBM(Binary((2,)), ReLU((1,)), randn(2,1))
    randn!(rbm.visible.θ)
    randn!(rbm.hidden.θ)
    rand!(rbm.hidden.γ)
    rbm.hidden.γ .+= 0.5

    v = sample_v_from_v(rbm, bitrand(size(rbm.visible)..., 1000); steps=100)
    F = free_energy(rbm, v)

    ω = @inferred weight_norms(rbm)
    @test ω ≈ [norm(rbm.w)]
    @inferred rescale_weights!(rbm)
    @test free_energy(rbm, v) ≈ F .- sum(log, ω)
end
