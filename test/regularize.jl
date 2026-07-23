import Zygote
using EllipsisNotation: (..)
using Random: bitrand
using RestrictedBoltzmannMachines: ∂free_energy
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

using FillArrays: Ones

@testset "∂regularize_fields with immutable layer parameters" begin
    # layers backed by lazy/read-only arrays must still get a mutable gradient buffer
    layer = Binary(Ones(1, 5))
    ∂ = ∂regularize_fields(layer; l2_fields = 0.5)
    @test ∂ ≈ fill(0.5, 1, 5)
end

using RestrictedBoltzmannMachines: RBM, Potts, sample_from_inputs, zerosum!

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
