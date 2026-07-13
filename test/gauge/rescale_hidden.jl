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
    @test rescale_hidden!(rbm, λ)

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

using RestrictedBoltzmannMachines: Spin, Potts, PottsGumbel, nsReLU

@testset "rescale_activations! is a no-op for layers without scale parameters" begin
    N = (3, 2)
    layers = (
        Binary(; θ = randn(N...)),
        Spin(; θ = randn(N...)),
        Potts(; θ = randn(N...)),
        PottsGumbel(; θ = randn(N...)),
        nsReLU(; θ = randn(N...), Δ = randn(N...), ξ = randn(N...)),
    )
    λ = 0.5 .+ rand(N...)
    for layer in layers
        par = copy(layer.par)
        @test !rescale_activations!(layer, λ)
        @test layer.par == par
    end
end

@testset "rescale_hidden! and rescale_weights! no-op for discrete hidden units" begin
    rbm = RBM(Binary(; θ = randn(3)), Binary(; θ = randn(2)), randn(3, 2))
    rbm_copy = deepcopy(rbm)
    @test !rescale_hidden!(rbm, 0.5 .+ rand(2))
    @test !rescale_weights!(rbm)
    @test rbm.visible.par == rbm_copy.visible.par
    @test rbm.hidden.par == rbm_copy.hidden.par
    @test rbm.w == rbm_copy.w
end

@testset "rescale_hidden! free energy invariance ($name hidden)" for (name, hidden) in [
    ("Gaussian", Gaussian(; θ = randn(2), γ = 0.5 .+ rand(2))),
    ("dReLU", dReLU(; θp = randn(2), θn = randn(2), γp = 0.5 .+ rand(2), γn = 0.5 .+ rand(2))),
    ("pReLU", pReLU(; θ = randn(2), γ = 0.5 .+ rand(2), Δ = randn(2), η = rand(2) .- 0.5)),
    ("xReLU", xReLU(; θ = randn(2), γ = 0.5 .+ rand(2), Δ = randn(2), ξ = randn(2))),
]
    rbm = RBM(Binary(; θ = randn(3)), hidden, randn(3, 2))
    v = bitrand(3, 100)
    F = free_energy(rbm, v)
    λ = 0.5 .+ rand(2)
    @test rescale_hidden!(rbm, λ)
    # free energies shift by the constant log-Jacobian of h -> h / λ
    @test free_energy(rbm, v) ≈ F .+ sum(log, λ)
end
