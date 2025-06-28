import Zygote
using RestrictedBoltzmannMachines: ∂cgf
using RestrictedBoltzmannMachines: cgfs
using RestrictedBoltzmannMachines: dReLU
using RestrictedBoltzmannMachines: energies
using RestrictedBoltzmannMachines: grad2ave
using RestrictedBoltzmannMachines: grad2var
using RestrictedBoltzmannMachines: mean_abs_from_inputs
using RestrictedBoltzmannMachines: mean_from_inputs
using RestrictedBoltzmannMachines: mode_from_inputs
using RestrictedBoltzmannMachines: nsReLU
using RestrictedBoltzmannMachines: var_from_inputs
using RestrictedBoltzmannMachines: xReLU
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
    @test mean_from_inputs(drelu) ≈ mean_from_inputs(xrelu)  ≈ mean_from_inputs(nrelu)
    @test mean_abs_from_inputs(drelu) ≈ mean_abs_from_inputs(xrelu) ≈ mean_abs_from_inputs(nrelu)
    @test var_from_inputs(drelu) ≈ var_from_inputs(xrelu) ≈ var_from_inputs(nrelu)
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
