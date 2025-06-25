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
using Statistics: mean
using Test: @test, @testset

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

    @test only(gs).visible.par ≈ ∂.visible
    @test only(gs).hidden.par ≈ ∂.hidden
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
    @test only(gs).visible.par ≈ ∂.visible
    @test isnothing(only(gs).hidden) && iszero(∂.hidden)
    @test only(gs).w ≈ ∂.w
end

@testset "∂regularize_fields" begin
    l2_fields = rand()

    layer = Binary(; θ = randn(3,5))
    gs = Zygote.gradient(layer) do layer
        l2_fields/2 * sum(abs2, layer.θ)
    end
    ∂ = ∂regularize_fields(layer; l2_fields)
    @test only(gs).par ≈ ∂

    layer = Gaussian(; θ = randn(3,5), γ = rand(3,5))
    gs = Zygote.gradient(layer) do layer
        l2_fields/2 * sum(abs2, layer.θ)
    end
    ∂ = ∂regularize_fields(layer; l2_fields)
    @test only(gs).par ≈ ∂
    @test iszero(∂[2, ..]) # ∂γ

    layer = dReLU(; θp = randn(3,5), θn = randn(3,5), γp = rand(3,5), γn = rand(3,5))
    gs = Zygote.gradient(layer) do layer
        l2_fields/2 * (sum(abs2, layer.θp) + sum(abs2, layer.θn))
    end
    ∂ = ∂regularize_fields(layer; l2_fields)
    @test only(gs).par ≈ ∂
    @test iszero(∂[3:4, ..]) # ∂γp, ∂γn

    layer = pReLU(; θ = randn(3,5), γ = rand(3,5), Δ = randn(3,5), η = rand(3,5) .- 0.5)
    gs = Zygote.gradient(layer) do layer
        l2_fields/2 * sum(abs2, layer.θ)
    end
    ∂ = ∂regularize_fields(layer; l2_fields)
    @test only(gs).par ≈ ∂
    @test iszero(∂[2:4, ..]) # ∂γ, ∂Δ, ∂η

    layer = xReLU(; θ = randn(3,5), γ = rand(3,5), Δ = randn(3,5), ξ = randn(3,5))
    gs = Zygote.gradient(layer) do layer
        l2_fields/2 * sum(abs2, layer.θ)
    end
    ∂ = ∂regularize_fields(layer; l2_fields)
    @test only(gs).par ≈ ∂
    @test iszero(∂[2:4, ..]) # ∂γ, ∂Δ, ∂ξ
end
