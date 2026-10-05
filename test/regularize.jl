import Zygote
using EllipsisNotation: (..)
using FillArrays: Ones
using Random: bitrand
using RestrictedBoltzmannMachines: ∂free_energy, ∂regularize!, ∂RBM, regularization_penalty
using RestrictedBoltzmannMachines: CompositeRegularizer, StandardizedParametersRegularizer,
    L2FieldsRegularizer, L1WeightsRegularizer, L2WeightsRegularizer, L2L1WeightsRegularizer
using RestrictedBoltzmannMachines: RBM, BinaryRBM, Binary, dReLU, Gaussian, nsReLU, pReLU, xReLU
using RestrictedBoltzmannMachines: free_energy, standardize
using Statistics: mean
using Test: @test, @testset

@testset "regularization penalties" begin
    rbm = BinaryRBM(randn(3, 5), randn(2), randn(3, 5, 2))
    λ = rand()
    @test regularization_penalty(rbm, L2FieldsRegularizer(λ)) ≈ λ / 2 * sum(abs2, rbm.visible.θ)
    @test regularization_penalty(rbm, L1WeightsRegularizer(λ)) ≈ λ * sum(abs, rbm.w)
    @test regularization_penalty(rbm, L2WeightsRegularizer(λ)) ≈ λ / 2 * sum(abs2, rbm.w)
    @test regularization_penalty(rbm, L2L1WeightsRegularizer(λ)) ≈
        λ / (2 * 15) * sum(abs2, sum(abs, rbm.w; dims = (1, 2)))
    @test regularization_penalty(rbm, CompositeRegularizer(L2FieldsRegularizer(λ), L2WeightsRegularizer(λ))) ≈
        regularization_penalty(rbm, L2FieldsRegularizer(λ)) + regularization_penalty(rbm, L2WeightsRegularizer(λ))
    @test iszero(regularization_penalty(rbm, CompositeRegularizer()))
    # on a plain RBM the standardized parameters are the parameters
    @test regularization_penalty(rbm, StandardizedParametersRegularizer(L2WeightsRegularizer(λ))) ==
        regularization_penalty(rbm, L2WeightsRegularizer(λ))
end

regularizers = (
    L2FieldsRegularizer(rand()),
    L1WeightsRegularizer(rand()),
    L2WeightsRegularizer(rand()),
    L2L1WeightsRegularizer(rand()),
    CompositeRegularizer(
        L2FieldsRegularizer(rand()), L1WeightsRegularizer(rand()),
        L2WeightsRegularizer(rand()), L2L1WeightsRegularizer(rand())
    ),
    CompositeRegularizer(L2WeightsRegularizer(rand()), CompositeRegularizer(L2FieldsRegularizer(rand()))), # nested
    StandardizedParametersRegularizer(L1WeightsRegularizer(rand())), # transparent on a plain RBM
    CompositeRegularizer(), # no regularization
)

@testset "∂regularize! $(nameof(typeof(reg)))" for reg in regularizers
    rbm = BinaryRBM(randn(3, 5), randn(3, 2), randn(3, 5, 3, 2))
    v = bitrand(3, 5, 100)

    gs = Zygote.gradient(rbm) do rbm
        mean(free_energy(rbm, v)) + regularization_penalty(rbm, reg)
    end

    ∂ = ∂free_energy(rbm, v)
    @test ∂regularize!(∂, rbm, reg) === ∂

    @test only(gs).visible.par ≈ ∂.visible
    @test only(gs).hidden.par ≈ ∂.hidden
    @test only(gs).w ≈ ∂.w
end

@testset "L2FieldsRegularizer on $(nameof(typeof(layer))) fields" for (layer, fields, other_rows) in (
        # (layer, field parameters, rows of `par` that get no regularization gradient)
        (Binary(; θ = randn(3, 5)), (:θ,), 2:1),
        (Gaussian(; θ = randn(3, 5), γ = rand(3, 5)), (:θ,), 2:2),
        (dReLU(; θp = randn(3, 5), θn = randn(3, 5), γp = rand(3, 5), γn = rand(3, 5)), (:θp, :θn), 3:4),
        (pReLU(; θ = randn(3, 5), γ = rand(3, 5), Δ = randn(3, 5), η = rand(3, 5) .- 0.5), (:θ,), 2:4),
        (xReLU(; θ = randn(3, 5), γ = rand(3, 5), Δ = randn(3, 5), ξ = randn(3, 5)), (:θ,), 2:4),
        (nsReLU(; θ = randn(3, 5), Δ = randn(3, 5), ξ = randn(3, 5)), (:θ,), 2:3),
    )
    λ = rand()
    rbm = RBM(layer, Binary(; θ = randn(2)), randn(3, 5, 2))
    gs = Zygote.gradient(rbm) do rbm
        λ / 2 * sum(sum(abs2, getproperty(rbm.visible, f)) for f in fields)
    end
    ∂ = ∂RBM(zero(rbm.visible.par), zero(rbm.hidden.par), zero(rbm.w))
    ∂regularize!(∂, rbm, L2FieldsRegularizer(λ))
    @test only(gs).visible.par ≈ ∂.visible
    @test iszero(∂.visible[other_rows, ..])
    @test iszero(∂.hidden)
    @test iszero(∂.w)
end

@testset "∂regularize! with immutable layer parameters" begin
    # the regularization gradient of a StandardizedRBM is accumulated in a fresh buffer,
    # which must be mutable even if the layer parameters are not
    rbm = standardize(RBM(Binary(Ones(1, 5)), Binary(; θ = randn(2)), randn(5, 2)))
    ∂ = ∂RBM(zeros(1, 5), zeros(1, 2), zeros(5, 2))
    ∂regularize!(∂, rbm, L2FieldsRegularizer(0.5))
    @test ∂.visible ≈ fill(0.5, 1, 5)
    @test iszero(∂.hidden)
    @test iszero(∂.w)
end
