import Random
using Test: @test, @testset, @inferred
using Statistics: mean, var
using Random: bitrand, rand!, randn!
using RestrictedBoltzmannMachines: RBM, Binary, visible, hidden, weights, free_energy
using RestrictedBoltzmannMachines: Gaussian, ReLU, dReLU, pReLU, xReLU
using RestrictedBoltzmannMachines: sample_v_from_v, sample_h_from_h, transfer_mean, var_from_inputs
using RestrictedBoltzmannMachines: rescale_hidden!, rescale_activations!

Random.seed!(23)

@testset "rescale_activations!" begin
    N = (7,2)
    layers = (
        Gaussian(randn(N...), randn(N...)),
        ReLU(randn(N...), randn(N...)),
        dReLU(randn(N...), randn(N...), randn(N...), randn(N...)),
        pReLU(randn(N...), randn(N...), randn(N...), 2rand(N...) .- 1),
        xReLU(randn(N...), randn(N...), randn(N...), randn(N...))
    )
    λ = rand(N...)
    for layer in layers
        μ = transfer_mean(layer)
        ν = var_from_inputs(layer)
        rescale_activations!(layer, λ)
        @test transfer_mean(layer) ≈ μ ./ λ
        @test var_from_inputs(layer) ≈ ν ./ λ.^2
    end
end

@testset "rescale_hidden!" begin
    rbm = RBM(Binary(2), ReLU(1), randn(2,1))
    randn!(rbm.visible.θ)
    randn!(rbm.hidden.θ)
    rand!(rbm.hidden.γ)
    rbm.hidden.γ .+= 0.5

    v = sample_v_from_v(rbm, bitrand(size(visible(rbm))..., 20000); steps=100)
    h = sample_h_from_h(rbm, rand(size(hidden(rbm))..., 20000); steps=100)
    ave_v = mean(v; dims=2)
    ave_h = mean(h; dims=2)
    var_v = var(v; dims=2)
    var_h = var(h; dims=2)
    F = free_energy(rbm, v)

    λ = [1 + rand()]
    rescale_hidden!(rbm, λ)

    @test free_energy(rbm, v) ≈ F .+ sum(log, λ)

    v = sample_v_from_v(rbm, bitrand(size(visible(rbm))..., 20000); steps=100)
    h = sample_h_from_h(rbm, rand(size(hidden(rbm))..., 20000); steps=100)

    @test mean(v; dims=2) ≈ ave_v rtol=0.1
    @test mean(h; dims=2) ≈ ave_h ./ λ rtol=0.1
    @test var(v; dims=2) ≈ var_v rtol=0.1
    @test var(h; dims=2) ≈ var_h ./ λ.^2 rtol=0.1
end
