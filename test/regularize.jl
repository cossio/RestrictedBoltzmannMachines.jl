import Zygote
using EllipsisNotation: (..)
using Random: bitrand
using RestrictedBoltzmannMachines: ∂free_energy
using RestrictedBoltzmannMachines: ∂regularize
using RestrictedBoltzmannMachines: ∂regularize_fields
using RestrictedBoltzmannMachines: ∂regularize!
using RestrictedBoltzmannMachines: Binary
using RestrictedBoltzmannMachines: BinaryRBM
using RestrictedBoltzmannMachines: dReLU
using RestrictedBoltzmannMachines: free_energy
using RestrictedBoltzmannMachines: Gaussian
using RestrictedBoltzmannMachines: pReLU
using RestrictedBoltzmannMachines: ReLU
using RestrictedBoltzmannMachines: xReLU
using RestrictedBoltzmannMachines: regularization_penalty
using Statistics: mean
using Test: @test, @testset

@testset "∂regularize!" begin
    rbm = BinaryRBM(randn(3, 5), randn(3, 2), randn(3, 5, 3, 2))
    v = bitrand(3, 5, 100)

    l2_fields = rand()
    l1_weights = rand()
    l2_weights = rand()
    l2l1_weights = rand()

    gs = Zygote.gradient(rbm) do rbm
        mean(free_energy(rbm, v)) + regularization_penalty(rbm; l2_fields, l1_weights, l2_weights, l2l1_weights)
    end

    ∂ = ∂free_energy(rbm, v)
    ∂regularize!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights)

    @test only(gs).visible.par ≈ ∂.visible
    @test only(gs).hidden.par ≈ ∂.hidden
    @test only(gs).w ≈ ∂.w
end

@testset "∂regularize" begin
    rbm = BinaryRBM(randn(3, 5), randn(3, 2), randn(3, 5, 3, 2))

    l2_fields = rand()
    l1_weights = rand()
    l2_weights = rand()
    l2l1_weights = rand()

    gs = Zygote.gradient(rbm) do rbm
        regularization_penalty(rbm; l2_fields, l1_weights, l2_weights, l2l1_weights)
    end

    ∂ = ∂regularize(rbm; l2_fields, l1_weights, l2_weights, l2l1_weights)
    @test only(gs).visible.par ≈ ∂.visible
    @test isnothing(only(gs).hidden) && iszero(∂.hidden)
    @test only(gs).w ≈ ∂.w
end

@testset "∂regularize_fields" begin
    l2_fields = rand()

    layer = Binary(; θ = randn(3, 5))
    gs = Zygote.gradient(layer) do layer
        l2_fields / 2 * sum(abs2, layer.θ)
    end
    ∂ = ∂regularize_fields(layer; l2_fields)
    @test only(gs).par ≈ ∂

    layer = Gaussian(; θ = randn(3, 5), γ = rand(3, 5))
    gs = Zygote.gradient(layer) do layer
        l2_fields / 2 * sum(abs2, layer.θ)
    end
    ∂ = ∂regularize_fields(layer; l2_fields)
    @test only(gs).par ≈ ∂
    @test iszero(∂[2, ..]) # ∂γ

    layer = dReLU(; θp = randn(3, 5), θn = randn(3, 5), γp = rand(3, 5), γn = rand(3, 5))
    gs = Zygote.gradient(layer) do layer
        l2_fields / 2 * (sum(abs2, layer.θp) + sum(abs2, layer.θn))
    end
    ∂ = ∂regularize_fields(layer; l2_fields)
    @test only(gs).par ≈ ∂
    @test iszero(∂[3:4, ..]) # ∂γp, ∂γn

    layer = pReLU(; θ = randn(3, 5), γ = rand(3, 5), Δ = randn(3, 5), η = rand(3, 5) .- 0.5)
    gs = Zygote.gradient(layer) do layer
        l2_fields / 2 * sum(abs2, layer.θ)
    end
    ∂ = ∂regularize_fields(layer; l2_fields)
    @test only(gs).par ≈ ∂
    @test iszero(∂[2:4, ..]) # ∂γ, ∂Δ, ∂η

    layer = xReLU(; θ = randn(3, 5), γ = rand(3, 5), Δ = randn(3, 5), ξ = randn(3, 5))
    gs = Zygote.gradient(layer) do layer
        l2_fields / 2 * sum(abs2, layer.θ)
    end
    ∂ = ∂regularize_fields(layer; l2_fields)
    @test only(gs).par ≈ ∂
    @test iszero(∂[2:4, ..]) # ∂γ, ∂Δ, ∂ξ
end

using RestrictedBoltzmannMachines: RBM, Potts, sample_from_inputs, zerosum!
using RestrictedBoltzmannMachines: prox_glasso!
using LinearAlgebra: norm

@testset "group-lasso gradients (Potts)" begin
    # ∂regularize! must match the autodiff gradient of regularization_penalty for the
    # grouped (Potts color axis) gl2l1 and glasso terms.
    rbm = RBM(Potts(; θ = randn(3, 4)), Binary(; θ = randn(2)), randn(3, 4, 2))
    v = sample_from_inputs(rbm.visible, zeros(3, 4, 20))

    l2_fields, l1_weights, l2_weights, l2l1_weights, gl2l1_weights, glasso_weights = rand(6)

    gs = Zygote.gradient(rbm) do rbm
        mean(free_energy(rbm, v)) + regularization_penalty(
            rbm; l2_fields, l1_weights, l2_weights, l2l1_weights, gl2l1_weights, glasso_weights
        )
    end

    ∂ = ∂free_energy(rbm, v)
    ∂regularize!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights, gl2l1_weights, glasso_weights)

    @test only(gs).visible.par ≈ ∂.visible
    @test only(gs).hidden.par ≈ ∂.hidden
    @test only(gs).w ≈ ∂.w

    # ∂regularize (non-mutating) agrees too
    ∂w = ∂regularize(rbm; gl2l1_weights, glasso_weights).w
    gw = Zygote.gradient(rbm -> regularization_penalty(rbm; gl2l1_weights, glasso_weights), rbm)
    @test only(gw).w ≈ ∂w
end

@testset "group-lasso reduces to l1/l2l1 for non-Potts" begin
    # For Binary (scalar groups): glasso == l1 and gl2l1 == l2l1, in both penalty and gradient.
    rbm = BinaryRBM(randn(3, 5), randn(3, 2), randn(3, 5, 3, 2))
    λ = rand()

    @test regularization_penalty(rbm; glasso_weights = λ) ≈ regularization_penalty(rbm; l1_weights = λ)
    @test regularization_penalty(rbm; gl2l1_weights = λ) ≈ regularization_penalty(rbm; l2l1_weights = λ)

    @test ∂regularize(rbm; glasso_weights = λ).w ≈ ∂regularize(rbm; l1_weights = λ).w
    @test ∂regularize(rbm; gl2l1_weights = λ).w ≈ ∂regularize(rbm; l2l1_weights = λ).w
end

@testset "prox_glasso! (Potts)" begin
    rbm = RBM(Potts(; θ = randn(3, 4)), Binary(; θ = randn(2)), randn(3, 4, 2))
    zerosum!(rbm)
    w0 = copy(rbm.w)
    t = 0.3

    prox_glasso!(rbm, t)

    # Block soft-threshold applied per (site, hidden) color group.
    for i in 1:4, μ in 1:2
        g = norm(w0[:, i, μ])
        expected = max(0, 1 - t / g) * w0[:, i, μ]
        @test rbm.w[:, i, μ] ≈ expected
        if g ≤ t
            @test all(iszero, rbm.w[:, i, μ]) # groups below threshold land on exact zero
        end
    end

    # prox preserves the zerosum gauge (scales each color group uniformly)
    @test all(abs.(sum(rbm.w; dims = 1)) .< 1.0e-10)
end

@testset "prox_glasso! zeros a small group and preserves it" begin
    rbm = RBM(Potts(; θ = randn(2, 3)), Binary(; θ = randn(2)), randn(2, 3, 2))
    # Make one group tiny; a threshold above its norm must zero it exactly.
    rbm.w[:, 1, 1] .= [1.0e-4, -1.0e-4]
    prox_glasso!(rbm, 0.01)
    @test all(iszero, rbm.w[:, 1, 1])
    # An already-zero group stays zero (no NaNs from the 0/0 guard).
    prox_glasso!(rbm, 0.01)
    @test all(iszero, rbm.w[:, 1, 1])
    @test all(isfinite, rbm.w)
end

@testset "prox_glasso! (Binary scalar groups)" begin
    # For non-Potts layers, prox_glasso! is a plain elementwise soft-threshold (L1 prox).
    rbm = BinaryRBM(randn(4), randn(3), randn(4, 3))
    w0 = copy(rbm.w)
    t = 0.25
    prox_glasso!(rbm, t)
    @test rbm.w ≈ sign.(w0) .* max.(0, abs.(w0) .- t)
end

@testset "∂regularize! zerosum keyword" begin
    # zerosum=true must be equivalent to regularizing and then projecting with zerosum!
    rbm = RBM(Potts(; θ = randn(3, 4)), Binary(; θ = randn(2)), randn(3, 4, 2))
    zerosum!(rbm)
    v = sample_from_inputs(rbm.visible, zeros(3, 4, 50))
    ∂ = ∂free_energy(rbm, v)
    ∂ref = deepcopy(∂)

    l2_fields, l1_weights, l2_weights, l2l1_weights = rand(4)
    ∂regularize!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights, zerosum = true)
    ∂regularize!(∂ref, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights)
    zerosum!(∂ref, rbm)

    @test ∂.visible ≈ ∂ref.visible
    @test ∂.hidden ≈ ∂ref.hidden
    @test ∂.w ≈ ∂ref.w
    # the projected weight gradient is in the zerosum gauge along the color dimension
    @test all(abs.(sum(∂.w; dims = 1)) .< 1.0e-10)
end
