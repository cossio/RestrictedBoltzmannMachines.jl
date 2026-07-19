import Zygote
using EllipsisNotation: (..)
using RestrictedBoltzmannMachines: ∂cgf, ∂regularize_fields
using RestrictedBoltzmannMachines: cgfs
using RestrictedBoltzmannMachines: dReLU
using RestrictedBoltzmannMachines: energies
using RestrictedBoltzmannMachines: grad2ave
using RestrictedBoltzmannMachines: grad2var
using RestrictedBoltzmannMachines: initialize!
using RestrictedBoltzmannMachines: batchmean
using RestrictedBoltzmannMachines: mean_abs_from_inputs
using RestrictedBoltzmannMachines: mean_from_inputs
using RestrictedBoltzmannMachines: meanvar_from_inputs
using RestrictedBoltzmannMachines: mode_from_inputs
using RestrictedBoltzmannMachines: nsReLU
using RestrictedBoltzmannMachines: sample_from_inputs
using RestrictedBoltzmannMachines: std_from_inputs
using RestrictedBoltzmannMachines: var_from_inputs
using RestrictedBoltzmannMachines: xReLU
using RestrictedBoltzmannMachines: shift_fields
using Statistics: mean
using Test: @inferred
using Test: @test
using Test: @testset

@testset "nsReLU convert" begin
    N = (10, 7)
    B = 13
    x = randn(N..., B)

    nrelu = nsReLU(; θ = randn(N...), Δ = randn(N...), ξ = randn(N...))
    drelu = @inferred dReLU(nrelu)
    xrelu = @inferred xReLU(nrelu)
    @test energies(drelu, x) ≈ energies(xrelu, x) ≈ energies(nrelu, x)
    @test all(isone, xrelu.γ)
    @test cgfs(drelu) ≈ cgfs(xrelu) ≈ cgfs(nrelu)
    @test mode_from_inputs(drelu) ≈ mode_from_inputs(xrelu) ≈ mode_from_inputs(nrelu)
    @test mean_from_inputs(drelu) ≈ mean_from_inputs(xrelu) ≈ mean_from_inputs(nrelu)
    @test mean_abs_from_inputs(drelu) ≈ mean_abs_from_inputs(xrelu) ≈ mean_abs_from_inputs(nrelu)
    @test var_from_inputs(drelu) ≈ var_from_inputs(xrelu) ≈ var_from_inputs(nrelu)

    # eltype is preserved (https://github.com/cossio/RestrictedBoltzmannMachines.jl/issues/119)
    nrelu32 = nsReLU(Float32, N)
    @test eltype(xReLU(nrelu32).par) == Float32
    @test eltype(dReLU(nrelu32).par) == Float32
    @test eltype(cgfs(nrelu32, zeros(Float32, N..., B))) == Float32
end

@testset "nsReLU" begin
    N = (3, 5)
    layer = nsReLU(; θ = randn(N...), Δ = randn(N...), ξ = randn(N...))
    gs = Zygote.gradient(layer) do layer
        sum(cgfs(layer))
    end
    ∂ = ∂cgf(layer)
    @test ∂ ≈ only(gs).par
    @test grad2ave(layer, ∂) ≈ mean_from_inputs(layer)
end

@testset "nsReLU constructors" begin
    layer = nsReLU(Float64, (3, 5))
    @test size(layer) == (3, 5)
    @test all(iszero, layer.θ)
    @test all(iszero, layer.Δ)
    @test all(iszero, layer.ξ)

    layer2 = nsReLU((4, 2))
    @test size(layer2) == (4, 2)
    @test eltype(layer2.θ) == Float64

    @test propertynames(layer) == (:θ, :Δ, :ξ)
end

@testset "nsReLU delegation" begin
    N = (3, 5)
    B = 7
    layer = nsReLU(; θ = randn(N...), Δ = randn(N...), ξ = randn(N...))
    inputs = randn(N..., B)
    xrelu = xReLU(layer)

    @test sample_from_inputs(layer, inputs) isa AbstractArray
    μ, ν = meanvar_from_inputs(layer, inputs)
    @test μ ≈ meanvar_from_inputs(xrelu, inputs)[1]
    @test ν ≈ meanvar_from_inputs(xrelu, inputs)[2]
    @test std_from_inputs(layer, inputs) ≈ sqrt.(var_from_inputs(layer, inputs))
end

@testset "nsReLU initialize!" begin
    N = (5,)
    layer = nsReLU(; θ = randn(N...), Δ = randn(N...), ξ = randn(N...))

    # initialize! without data
    initialize!(layer)
    @test all(iszero, layer.θ)
    @test all(iszero, layer.Δ)
    @test all(iszero, layer.ξ)

    # initialize! with data
    layer2 = nsReLU(; θ = randn(N...), Δ = randn(N...), ξ = randn(N...))
    data = 0.5randn(N..., 10^5) .+ 1
    initialize!(layer2, data)
    @test layer2.θ ≈ batchmean(layer2, data) rtol = 0.05
    @test all(iszero, layer2.Δ)
    @test all(iszero, layer2.ξ)
end

@testset "nsReLU ∂regularize_fields" begin
    layer = nsReLU(; θ = randn(3, 5), Δ = randn(3, 5), ξ = randn(3, 5))
    ∂ = ∂regularize_fields(layer; l2_fields = 0.1)
    @test size(∂) == size(layer.par)
    @test ∂[1, ..] ≈ 0.1 * layer.θ
    @test all(iszero, ∂[2, ..])
    @test all(iszero, ∂[3, ..])
end

@testset "nsReLU shift_fields" begin
    layer = nsReLU(; θ = randn(3, 5), Δ = randn(3, 5), ξ = randn(3, 5))
    a = randn(3, 5)
    shifted = shift_fields(layer, a)
    @test shifted isa nsReLU
    @test shifted.θ ≈ layer.θ .+ a
    @test shifted.Δ ≈ layer.Δ
    @test shifted.ξ ≈ layer.ξ
end
