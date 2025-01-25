import Random
import Zygote
using Base: tail
using EllipsisNotation: (..)
using LogExpFunctions: logistic
using QuadGK: quadgk
using Random: bitrand
using Random: rand!
using Random: randn!
using RestrictedBoltzmannMachines: ∂cgf
using RestrictedBoltzmannMachines: ∂energy
using RestrictedBoltzmannMachines: ∂free_energy
using RestrictedBoltzmannMachines: batchdims
using RestrictedBoltzmannMachines: Binary
using RestrictedBoltzmannMachines: binary_rand
using RestrictedBoltzmannMachines: cgf
using RestrictedBoltzmannMachines: cgfs
using RestrictedBoltzmannMachines: energies
using RestrictedBoltzmannMachines: energy
using RestrictedBoltzmannMachines: free_energy
using RestrictedBoltzmannMachines: gauss_energy
using RestrictedBoltzmannMachines: grad2ave
using RestrictedBoltzmannMachines: mean_abs_from_inputs
using RestrictedBoltzmannMachines: mean_from_inputs
using RestrictedBoltzmannMachines: meanvar_from_inputs
using RestrictedBoltzmannMachines: mode_from_inputs
using RestrictedBoltzmannMachines: Potts
using RestrictedBoltzmannMachines: PottsGumbel
using RestrictedBoltzmannMachines: RBM
using RestrictedBoltzmannMachines: sample_from_inputs
using RestrictedBoltzmannMachines: sample_v_from_v
using RestrictedBoltzmannMachines: std_from_inputs
using RestrictedBoltzmannMachines: var_from_inputs
using RestrictedBoltzmannMachines: vstack
using Statistics: cov
using Statistics: mean
using Statistics: var
using Test: @inferred
using Test: @test
using Test: @testset

Random.seed!(2)

_layers = (
    Potts,
    PottsGumbel,
)

@testset "testing $Layer" for Layer in _layers
    sz = (3,2)
    layer = Layer(sz)
    randn!(layer.par)

    @test (@inferred size(layer)) == sz
    for (d, n) in enumerate(sz)
        @test (@inferred size(layer, d)) == n
    end

    @test (@inferred length(layer)) == prod(sz)
    @test (@inferred ndims(layer)) == length(sz)
    @test (@inferred energy(layer, rand(sz...))) isa Number
    @test (@inferred cgf(layer)) isa Number
    @test (@inferred cgf(layer, rand(sz...))) isa Number
    @test size(@inferred mean_from_inputs(layer)) == size(layer)
    @test size(@inferred var_from_inputs(layer)) == size(layer)
    @test size(@inferred sample_from_inputs(layer)) == size(layer)
    @test cgfs(layer, 0) ≈ cgfs(layer)
    @test std_from_inputs(layer) ≈ sqrt.(var_from_inputs(layer))

    if (layer isa Potts) || (layer isa PottsGumbel)
        @test size(@inferred cgfs(layer)) == (1, tail(size(layer))...)
    else
        @test size(@inferred cgfs(layer)) == size(layer)
    end

    @test size(@inferred sample_from_inputs(layer, 0)) == size(layer)

    for B in ((), (2,), (1,2))
        x = rand(sz..., B...)
        @test (@inferred batchdims(layer, x)) == (length(sz) + 1):ndims(x)
        @test size(@inferred energy(layer, x)) == (B...,)
        @test size(@inferred cgf(layer, x)) == (B...,)
        @test size(@inferred energies(layer, x)) == size(x)
        @test size(@inferred sample_from_inputs(layer, x)) == size(x)
        @test size(@inferred mean_from_inputs(layer, x)) == size(x)
        @test size(@inferred var_from_inputs(layer, x)) == size(x)
        @test @inferred(std_from_inputs(layer, x)) ≈ sqrt.(var_from_inputs(layer, x))
        @test all(energy(layer, mode_from_inputs(layer)) .≤ energy(layer, x))

        μ, ν = meanvar_from_inputs(layer, x)
        @test μ ≈ mean_from_inputs(layer, x)
        @test ν ≈ var_from_inputs(layer, x)

        if (layer isa Potts) || (layer isa PottsGumbel)
            @test size(@inferred cgfs(layer, x)) == (1, tail(size(x))...)
        else
            @test size(@inferred cgfs(layer, x)) == size(x)
        end

        if B == ()
            @test @inferred(energy(layer, x)) ≈ sum(energies(layer, x))
            @test @inferred(cgf(layer, x)) ≈ sum(cgfs(layer, x))
        else
            @test @inferred(energy(layer, x)) ≈ reshape(sum(energies(layer, x); dims=1:ndims(layer)), B)
            @test @inferred(cgf(layer, x)) ≈ reshape(sum(cgfs(layer, x); dims=1:ndims(layer)), B)
        end

        μ = @inferred mean_from_inputs(layer, x)
        @test only(Zygote.gradient(j -> sum(cgfs(layer, j)), x)) ≈ μ
    end

    ∂Γ = @inferred ∂cgf(layer)
    gs = Zygote.gradient(layer) do layer
        cgf(layer)
    end
    @test ∂Γ ≈ only(gs).par

    samples = @inferred sample_from_inputs(layer, zeros(size(layer)..., 10^6))
    @test @inferred(mean_from_inputs(layer)) ≈ reshape(mean(samples; dims=3), size(layer)) rtol=0.1 atol=0.01
    @test @inferred(var_from_inputs(layer)) ≈ reshape(var(samples; dims=ndims(samples)), size(layer)) rtol=0.1
    @test @inferred(mean_abs_from_inputs(layer)) ≈ reshape(mean(abs.(samples); dims=ndims(samples)), size(layer)) rtol=0.1

    ∂Γ = @inferred ∂cgf(layer)
    ∂E = @inferred ∂energy(layer, samples)
    @test ∂Γ ≈ -∂E rtol=0.1

    gs = Zygote.gradient(layer) do layer
        sum(energies(layer, samples)) / size(samples)[end]
    end
    @test ∂E ≈ only(gs).par
end

@testset "discrete layers ($Layer)" for Layer in (Potts, PottsGumbel)
    N = (3, 4, 5)
    B = 13
    layer = Layer(; θ = randn(N...))
    x = bitrand(N..., B)
    @test energies(layer, x) ≈ -layer.θ .* x
    gs = Zygote.gradient(layer) do layer
        sum(cgfs(layer))
    end
    @test ∂cgf(layer) ≈ only(gs).par ≈ vstack((mean_from_inputs(layer),))
end

@testset "PottsGumbel" begin
    q = 3
    N = (4, 5)
    layer = PottsGumbel(; θ = randn(q, N...))
    @test cgfs(layer) ≈ log.(sum(exp.(layer.θ[h:h,:,:,:]) for h in 1:q))
    @test all(sum(mean_from_inputs(layer); dims=1) .≈ 1)
    # samples are proper one-hot
    @test sort(unique(sample_from_inputs(layer))) == [0, 1]
    @test all(sum(sample_from_inputs(layer); dims=1) .== 1)

    gs = Zygote.gradient(layer) do layer
        sum(cgfs(layer))
    end
    ∂ = ∂cgf(layer)
    @test ∂ ≈ only(gs).par ≈ vstack((mean_from_inputs(layer),))
    @test grad2ave(layer, ∂) ≈ mean_from_inputs(layer)
end

@testset "grad2ave $Layer" for Layer in _layers
    layer = Layer((5,))
    rbm = RBM(layer, Binary(; θ = randn(3)), randn(5,3))
    v = sample_v_from_v(rbm, randn(5,100); steps=100)
    ∂ = ∂free_energy(rbm, v)
    @test (@inferred grad2ave(rbm.visible, -∂.visible)) ≈ dropdims(mean(v; dims=2); dims=2)
end
