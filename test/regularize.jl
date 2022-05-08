import RestrictedBoltzmannMachines as RBMs
import Zygote
using Test: @test, @testset
using Statistics: mean
using Random: bitrand
using RestrictedBoltzmannMachines: BinaryRBM, free_energy, ∂free_energy
using RestrictedBoltzmannMachines: Binary, Gaussian, ReLU, dReLU, pReLU, xReLU
using RestrictedBoltzmannMachines: ∂regularize!, ∂regularize, ∂regularize_fields

@testset "∂regularize!" begin
    rbm = BinaryRBM(randn(3,5), randn(3,2), randn(3,5,3,2))
    v = bitrand(3,5,100)
    vdims = ntuple(identity, ndims(rbm.visible))
    N = length(rbm.visible)

    l2_fields = rand()
    l1_weights = rand()
    l2_weights = rand()
    l2l1_weights = rand()

    gs = Zygote.gradient(rbm) do rbm
        F = mean(free_energy(rbm, v))
        L2_fields = sum(abs2, rbm.visible.θ)
        L1_weights = sum(abs, rbm.w)
        L2_weights = sum(abs2, rbm.w)
        L2L1_weights = sum(abs2, sum(abs, rbm.w; dims=vdims))
        return (
            F + l2_fields/2 * L2_fields + l1_weights * L1_weights +
            l2_weights/2 * L2_weights +
            l2l1_weights/(2N) * L2L1_weights
        )
    end

    ∂ = ∂free_energy(rbm, v)
    ∂regularize!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights)

    @test only(gs).visible.θ ≈ ∂.visible.θ
    @test only(gs).hidden.θ ≈ ∂.hidden.θ
    @test only(gs).w ≈ ∂.w
end

@testset "∂regularize" begin
    rbm = BinaryRBM(randn(3,5), randn(3,2), randn(3,5,3,2))
    dims = ntuple(identity, ndims(rbm.visible))
    N = length(rbm.visible)

    l2_fields = rand()
    l1_weights = rand()
    l2_weights = rand()
    l2l1_weights = rand()

    gs = Zygote.gradient(rbm) do rbm
        return (
            l2_fields/2 * sum(abs2, rbm.visible.θ) +
            l1_weights * sum(abs, rbm.w) +
            l2_weights/2 * sum(abs2, rbm.w) +
            l2l1_weights/(2N) * sum(abs2, sum(abs, rbm.w; dims))
        )
    end

    ∂ = ∂regularize(rbm; l2_fields, l1_weights, l2_weights, l2l1_weights)

    @test only(gs).visible.θ ≈ ∂.visible.θ
    @test isnothing(only(gs).hidden) && isnothing(∂.hidden)
    @test only(gs).w ≈ ∂.w
end

@testset "∂regularize_fields" begin
    l2_fields = rand()

    layer = Binary(randn(3,5))
    gs = Zygote.gradient(layer) do layer
        l2_fields/2 * sum(abs2, layer.θ)
    end
    ∂ = ∂regularize_fields(layer; l2_fields)
    @test only(gs).θ ≈ ∂.θ

    layer = Gaussian(randn(3,5), rand(3,5))
    gs = Zygote.gradient(layer) do layer
        l2_fields/2 * sum(abs2, layer.θ)
    end
    ∂ = ∂regularize_fields(layer; l2_fields)
    @test only(gs).θ ≈ ∂.θ
    @test isnothing(only(gs).γ) && iszero(∂.γ)

    layer = dReLU(randn(3,5), randn(3,5), rand(3,5), rand(3,5))
    gs = Zygote.gradient(layer) do layer
        l2_fields/2 * (sum(abs2, layer.θp) + sum(abs2, layer.θn))
    end
    ∂ = ∂regularize_fields(layer; l2_fields)
    @test only(gs).θp ≈ ∂.θp
    @test only(gs).θn ≈ ∂.θn
    @test isnothing(only(gs).γp) && iszero(∂.γp)
    @test isnothing(only(gs).γn) && iszero(∂.γn)

    layer = pReLU(randn(3,5), rand(3,5), randn(3,5), rand(3,5) .- 0.5)
    gs = Zygote.gradient(layer) do layer
        l2_fields/2 * sum(abs2, layer.θ)
    end

    ∂ = ∂regularize_fields(layer; l2_fields)
    @test only(gs).θ ≈ ∂.θ
    @test isnothing(only(gs).γ) && iszero(∂.γ)
    @test isnothing(only(gs).Δ) && iszero(∂.Δ)
    @test isnothing(only(gs).η) && iszero(∂.η)

    layer = xReLU(randn(3,5), rand(3,5), randn(3,5), randn(3,5))
    gs = Zygote.gradient(layer) do layer
        l2_fields/2 * sum(abs2, layer.θ)
    end

    ∂ = ∂regularize_fields(layer; l2_fields)
    @test only(gs).θ ≈ ∂.θ
    @test isnothing(only(gs).γ) && iszero(∂.γ)
    @test isnothing(only(gs).Δ) && iszero(∂.Δ)
    @test isnothing(only(gs).ξ) && iszero(∂.ξ)
end
