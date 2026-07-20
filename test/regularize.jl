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
using RestrictedBoltzmannMachines: prox_glasso!, CenteredRBM, StandardizedRBM, pcd!
using LinearAlgebra: norm
using Random: seed!

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

@testset "regularization_penalty is NaN-safe at zero weight groups" begin
    # A Zygote gradient through regularization_penalty must stay finite when a Potts color
    # group is exactly zero (e.g. one already pruned by prox_glasso!), both when gl2l1/glasso
    # are unused (the sqrt(0) singularity must not leak into unrelated terms via 0 * Inf) and
    # when they are the active regularizer (the adjoint must pick the same w=0 subgradient as
    # ∂regularize!/∂glasso_weights/∂gl2l1_weights).
    rbm = RBM(Potts(; θ = randn(3, 4)), Binary(; θ = randn(2)), randn(3, 4, 2))
    rbm.w[:, 1, 1] .= 0 # one exact-zero color group

    l2_fields, l1_weights, l2_weights, l2l1_weights = rand(4)

    gs0 = only(
        Zygote.gradient(rbm) do rbm
            regularization_penalty(rbm; l2_fields, l1_weights, l2_weights, l2l1_weights)
        end
    )
    @test all(isfinite, gs0.w)
    @test all(isfinite, gs0.visible.par)

    gl2l1_weights, glasso_weights = rand(2)
    gs = only(Zygote.gradient(rbm -> regularization_penalty(rbm; gl2l1_weights, glasso_weights), rbm))
    @test all(isfinite, gs.w)
    @test gs.w ≈ ∂regularize(rbm; gl2l1_weights, glasso_weights).w
    @test all(iszero, gs.w[:, 1, 1]) # zero group contributes zero subgradient
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

# Weight-gradient increment added by ∂regularize! (isolates the regularization term).
function _reg_w_increment(rbm, v; kw...)
    ∂ = ∂free_energy(rbm, v)
    w0 = copy(∂.w)
    ∂regularize!(∂, rbm; kw...)
    return ∂.w - w0
end

@testset "group-lasso gradients (CenteredRBM)" begin
    # With trivial (zero) offsets, the group-lasso weight gradient matches the plain RBM.
    rbm = RBM(Potts(; θ = randn(3, 4)), Binary(; θ = randn(2)), randn(3, 4, 2))
    zerosum!(rbm)
    crbm = CenteredRBM(rbm)
    v = sample_from_inputs(rbm.visible, zeros(3, 4, 20))
    gl2l1_weights, glasso_weights = rand(2)

    @test _reg_w_increment(crbm, v; gl2l1_weights, glasso_weights) ≈
        _reg_w_increment(rbm, v; gl2l1_weights, glasso_weights)

    # prox_glasso! on the CenteredRBM matches prox on its underlying weights.
    w0 = copy(crbm.w)
    prox_glasso!(crbm, 0.3)
    ref = RBM(Potts(; θ = randn(3, 4)), Binary(; θ = randn(2)), copy(w0))
    prox_glasso!(ref, 0.3)
    @test crbm.w ≈ ref.w
end

@testset "group-lasso gradients (StandardizedRBM)" begin
    # With trivial standardization (unit scale, zero offset), everything matches the plain RBM.
    rbm = RBM(Potts(; θ = randn(3, 4)), Binary(; θ = randn(2)), randn(3, 4, 2))
    zerosum!(rbm)
    srbm = StandardizedRBM(rbm)
    v = sample_from_inputs(rbm.visible, zeros(3, 4, 20))
    gl2l1_weights, glasso_weights = rand(2)

    incr_plain = _reg_w_increment(rbm, v; gl2l1_weights, glasso_weights)
    @test _reg_w_increment(srbm, v; gl2l1_weights, glasso_weights) ≈ incr_plain
    # regularize_unstandardized = false forwards to the underlying RBM.
    @test _reg_w_increment(srbm, v; gl2l1_weights, glasso_weights, regularize_unstandardized = false) ≈ incr_plain

    # prox_glasso! matches the plain RBM prox under trivial standardization.
    w0 = copy(srbm.w)
    prox_glasso!(srbm, 0.3)
    ref = RBM(Potts(; θ = randn(3, 4)), Binary(; θ = randn(2)), copy(w0))
    prox_glasso!(ref, 0.3)
    @test srbm.w ≈ ref.w
end

@testset "prox_glasso! (StandardizedRBM, nontrivial scale)" begin
    # Group norms are taken on the *unstandardized* weights; a group whose unstandardized
    # norm falls below t is zeroed, and the standardization factor cancels in the round-trip.
    rbm = RBM(Potts(; θ = randn(2, 3)), Binary(; θ = randn(2)), randn(2, 3, 2))
    zerosum!(rbm)
    scale_v = rand(2, 3) .+ 0.5
    scale_h = rand(2) .+ 0.5
    offset_v = zeros(2, 3)
    offset_h = zeros(2)
    srbm = StandardizedRBM(rbm, offset_v, offset_h, scale_v, scale_h)

    w0 = copy(srbm.w)
    # unstandardized weight = w ./ (scale_v ⊗ scale_h)
    cv = reshape(scale_v, (2, 3, 1))
    ch = reshape(scale_h, (1, 1, 2))
    w_unstd = w0 ./ (cv .* ch)
    t = 0.4
    prox_glasso!(srbm, t)

    for i in 1:3, μ in 1:2
        g = norm(w_unstd[:, i, μ])
        @test srbm.w[:, i, μ] ≈ max(0, 1 - t / g) * w0[:, i, μ]
    end
    @test all(isfinite, srbm.w)
end

@testset "pcd! proximal glasso ends in gauge with exact zeros" begin
    # Realistic case: Potts visible + continuous (Gaussian) hidden, rescale + zerosum on.
    # The proximal step runs before the gauge resets, so the model still ends in the
    # visible zero-sum gauge, and rescale_weights! renormalizes the surviving groups while
    # leaving zeroed groups zero.
    seed!(0)
    rbm = RBM(Potts(; θ = randn(3, 6)), Gaussian(; θ = randn(4), γ = ones(4)), 0.1 * randn(3, 6, 4))
    data = sample_from_inputs(rbm.visible, zeros(3, 6, 200))
    pcd!(rbm, data; iters = 40, batchsize = 32, glasso_weights = 0.5, rescale = true, zerosum = true)

    @test all(isfinite, rbm.w)
    @test maximum(abs, sum(rbm.w; dims = 1)) < 1.0e-8       # visible zero-sum gauge preserved
    groupnorms = vec(sqrt.(sum(abs2, rbm.w; dims = 1)))
    @test any(iszero, groupnorms)                            # exact-zero color groups survive
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
