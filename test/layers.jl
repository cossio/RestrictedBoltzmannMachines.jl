import Random
import Zygote
import RestrictedBoltzmannMachines as RBMs

using Test: @test, @testset
using Statistics: mean, var, cov
using Random: bitrand
using LogExpFunctions: logistic
using QuadGK: quadgk
using RestrictedBoltzmannMachines: flatten, batch_size, batchmean, batchvar, batchcov
using RestrictedBoltzmannMachines: Binary, Spin, Potts, Gaussian, ReLU, dReLU, xReLU, pReLU
using RestrictedBoltzmannMachines: transfer_mean, transfer_var, transfer_meanvar,
    transfer_std, transfer_mean_abs, transfer_sample, transfer_mode
using RestrictedBoltzmannMachines: energy, free_energy, free_energies, energies

Random.seed!(2)

_layers = (
    Binary,
    Spin,
    Potts,
    Gaussian,
    ReLU,
    dReLU,
    pReLU,
    xReLU
)

random_layer(::Type{T}, N::Int...) where {T <: Union{Binary,Spin,Potts}} = T(randn(N...))
random_layer(::Type{T}, N::Int...) where {T <: Union{Gaussian,ReLU}} = T(randn(N...), rand(N...))
random_layer(::Type{dReLU}, N::Int...) = dReLU(randn(N...), randn(N...), rand(N...), rand(N...))
random_layer(::Type{xReLU}, N::Int...) = xReLU(randn(N...), rand(N...), randn(N...), randn(N...))
random_layer(::Type{pReLU}, N::Int...) = pReLU(randn(N...), rand(N...), randn(N...), 2rand(N...) .- 1)

@testset "testing $Layer" for Layer in _layers
    N = (3,2)
    layer = random_layer(Layer, N...)

    @test (@inferred size(layer)) == N
    for d in 1:length(N)
        @test (@inferred size(layer, d)) == N[d]
    end
    @test (@inferred length(layer)) == prod(N)
    @test (@inferred ndims(layer)) == length(N)
    @test size(@inferred repeat(layer,2,3,4)) == (((2, 3) .* N)..., 4)
    @test (@inferred batch_size(layer, rand(N...))) == ()
    @test (@inferred energy(layer, rand(N...))) isa Number
    @test (@inferred free_energy(layer)) isa Number
    @test (@inferred free_energy(layer, rand(N...))) isa Number
    @test size(@inferred transfer_mean(layer)) == size(layer)
    @test size(@inferred transfer_var(layer)) == size(layer)
    @test size(@inferred transfer_sample(layer)) == size(layer)
    @test free_energies(layer, 0) ≈ free_energies(layer)
    @test transfer_std(layer) ≈ sqrt.(transfer_var(layer))

    if layer isa RBMs.Potts
        @test size(@inferred free_energies(layer)) == (1, size(layer)[2:end]...)
    else
        @test size(@inferred free_energies(layer)) == size(layer)
    end

    @test size(@inferred transfer_sample(layer, 0)) == size(layer)

    for B in ((), (2,), (1,2))
        x = rand(N..., B...)
        @test (@inferred batch_size(layer, x)) == (B...,)
        @test (@inferred RBMs.batchdims(layer, x)) == (length(N) + 1):ndims(x)
        @test size(@inferred energy(layer, x)) == (B...,)
        @test size(@inferred free_energy(layer, x)) == (B...,)
        @test size(@inferred energies(layer, x)) == size(x)
        @test size(@inferred transfer_sample(layer, x)) == size(x)
        @test size(@inferred transfer_mean(layer, x)) == size(x)
        @test size(@inferred transfer_var(layer, x)) == size(x)
        @test @inferred(transfer_std(layer, x)) ≈ sqrt.(transfer_var(layer, x))
        @test all(energy(layer, transfer_mode(layer)) .≤ energy(layer, x))

        μ, ν = transfer_meanvar(layer, x)
        @test μ ≈ transfer_mean(layer, x)
        @test ν ≈ transfer_var(layer, x)

        if layer isa RBMs.Potts
            @test size(@inferred free_energies(layer, x)) == (1, size(x)[2:end]...)
        else
            @test size(@inferred free_energies(layer, x)) == size(x)
        end

        if B == ()
            @test @inferred(energy(layer, x)) ≈ sum(energies(layer, x))
            @test @inferred(free_energy(layer, x)) ≈ sum(free_energies(layer, x))
            @test @inferred(batchmean(layer, x)) ≈ x
            @test @inferred(batchvar(layer, x)) == zeros(N)
            @test @inferred(batchcov(layer, x)) == zeros(N..., N...)
        else
            @test @inferred(energy(layer, x)) ≈ reshape(sum(energies(layer, x); dims=1:ndims(layer)), B)
            @test @inferred(free_energy(layer, x)) ≈ reshape(sum(free_energies(layer, x); dims=1:ndims(layer)), B)
            @test @inferred(batchmean(layer, x)) ≈ reshape(mean(x; dims=(ndims(layer) + 1):ndims(x)), N)
            @test @inferred(batchvar(layer, x)) ≈ reshape(var(x; dims=(ndims(layer) + 1):ndims(x), corrected=false), N)
            @test @inferred(batchcov(layer, x)) ≈ reshape(cov(flatten(layer, x); dims=2, corrected=false), N..., N...)
        end

        μ = @inferred transfer_mean(layer, x)
        @test only(Zygote.gradient(j -> sum(free_energies(layer, j)), x)) ≈ -μ
    end

    ∂F = @inferred RBMs.∂free_energy(layer)
    gs = Zygote.gradient(layer) do layer
        sum(free_energies(layer))
    end
    for (ω, ∂ω) in pairs(∂F)
        @test ∂ω ≈ getproperty(only(gs), ω)
    end

    samples = @inferred transfer_sample(layer, zeros(size(layer)..., 10^6))
    @test @inferred(transfer_mean(layer)) ≈ reshape(mean(samples; dims=3), size(layer)) rtol=0.1 atol=0.01
    @test @inferred(transfer_var(layer)) ≈ reshape(var(samples; dims=ndims(samples)), size(layer)) rtol=0.1
    @test @inferred(transfer_mean_abs(layer)) ≈ reshape(mean(abs.(samples); dims=ndims(samples)), size(layer)) rtol=0.1

    ∂F = @inferred RBMs.∂free_energy(layer)
    ∂E = @inferred RBMs.∂energy(layer, samples)
    @test length(∂F) == length(∂E)
    @test propertynames(∂F) == propertynames(∂E)
    for (∂f, ∂e) in zip(∂F, ∂E)
        @test ∂f ≈ ∂e rtol=0.1
    end

    gs = Zygote.gradient(layer) do layer
        sum(energies(layer, samples)) / size(samples)[end]
    end
    for (ω, ∂ω) in pairs(∂E)
        @test ∂ω ≈ getproperty(only(gs), ω)
    end
end

@testset "discrete layers ($Layer)" for Layer in (Binary, Spin, RBMs.Potts)
    N = (3, 4, 5)
    B = 13
    layer = Layer(randn(N...))
    x = bitrand(N..., B)
    @test energies(layer, x) ≈ -layer.θ .* x
    gs = Zygote.gradient(layer) do layer
        sum(free_energies(layer))
    end
    @test RBMs.∂free_energy(layer).θ ≈ only(gs).θ ≈ -transfer_mean(layer)
end

@testset "Binary" begin
    @testset "binary_rand" begin
        for θ in -5:5, u in 0.0:0.1:1.0
            @test (@. u * (1 + exp(-θ)) < 1) == @inferred RBMs.binary_rand(θ, u)
        end
        θ = randn(1000)
        u = rand(1000)
        @test RBMs.binary_rand.(θ, u) == (@. u * (1 + exp(-θ)) < 1)
    end

    layer = Binary(randn(7, 4, 5))
    transfer_var(layer) ≈ @. logistic(layer.θ) * logistic(-layer.θ)
    @test free_energies(layer) ≈ -log.(sum(exp.(layer.θ .* h) for h in 0:1))
    @test sort(unique(transfer_sample(layer))) == [0, 1]

    gs = Zygote.gradient(layer) do layer
        sum(free_energies(layer))
    end
    ∂ = RBMs.∂free_energy(layer)
    @test ∂.θ ≈ only(gs).θ ≈ -transfer_mean(layer)
    @test RBMs.grad2mean(layer, ∂) ≈ transfer_mean(layer)
    @test RBMs.grad2var(layer, ∂) ≈ transfer_var(layer)
end

@testset "Spin" begin
    layer = Spin(randn(7, 4, 5))
    @test free_energies(layer) ≈ -log.(sum(exp.(layer.θ .* h) for h in (-1, 1)))
    @test sort(unique(transfer_sample(layer))) == [-1, 1]

    gs = Zygote.gradient(layer) do layer
        sum(free_energies(layer))
    end
    ∂ = RBMs.∂free_energy(layer)
    @test ∂.θ ≈ only(gs).θ ≈ -transfer_mean(layer)
    @test RBMs.grad2mean(layer, ∂) ≈ transfer_mean(layer)
    @test RBMs.grad2var(layer, ∂) ≈ transfer_var(layer)
end

@testset "Potts" begin
    q = 3
    N = (4, 5)
    layer = RBMs.Potts(randn(q, N...))
    @test free_energies(layer) ≈ -log.(sum(exp.(layer.θ[h:h,:,:,:]) for h in 1:q))
    @test all(sum(transfer_mean(layer); dims=1) .≈ 1)
    # samples are proper one-hot
    @test sort(unique(transfer_sample(layer))) == [0, 1]
    @test all(sum(transfer_sample(layer); dims=1) .== 1)

    gs = Zygote.gradient(layer) do layer
        sum(free_energies(layer))
    end
    ∂ = RBMs.∂free_energy(layer)
    @test ∂.θ ≈ only(gs).θ ≈ -transfer_mean(layer)
    @test RBMs.grad2mean(layer, ∂) ≈ transfer_mean(layer)
    @test RBMs.grad2var(layer, ∂) ≈ transfer_var(layer)
end

@testset "Gaussian" begin
    N = (3, 4, 5)
    B = 7

    # bound γ away from zero to avoid numerical issues with quadgk
    layer = Gaussian(randn(N...), rand(N...) .+ 0.5)

    x = randn(size(layer)..., B)
    @test energies(layer, x) ≈ @. abs(layer.γ) * x^2 / 2 - layer.θ * x

    function quad_free(θ::Real, γ::Real)
        Z, ϵ = quadgk(h -> exp(-RBMs.gauss_energy(θ, γ, h)), -Inf,  Inf)
        return -log(Z)
    end

    @test free_energies(layer) ≈ quad_free.(layer.θ, layer.γ) rtol=1e-6

    gs = Zygote.gradient(layer) do layer
        sum(free_energies(layer))
    end
    μ = transfer_mean(layer)
    ν = transfer_var(layer)
    μ2 = @. ν + μ^2
    ∂ = RBMs.∂free_energy(layer)
    @test ∂.θ ≈ only(gs).θ ≈ -μ
    @test ∂.γ ≈ only(gs).γ ≈ μ2/2
    @test RBMs.grad2mean(layer, ∂) ≈ transfer_mean(layer)
    @test RBMs.grad2var(layer, ∂) ≈ transfer_var(layer)
end

@testset "ReLU" begin
    N = (3, 4, 5)
    B = 7

    # bound γ away from zero to avoid numerical issues with quadgk
    layer = ReLU(randn(N...), rand(N...) .+ 0.5)

    x = abs.(randn(size(layer)..., B))
    @test energies(layer, x) ≈ energies(Gaussian(layer.θ, layer.γ), x)

    function quad_free(θ::Real, γ::Real)
        Z, ϵ = quadgk(h -> exp(-RBMs.relu_energy(θ, γ, h)), 0,  Inf)
        return -log(Z)
    end
    @test free_energies(layer) ≈ @. quad_free(layer.θ, layer.γ)

    gs = Zygote.gradient(layer) do layer
        sum(free_energies(layer))
    end
    μ = transfer_mean(layer)
    ν = transfer_var(layer)
    μ2 = @. ν + μ^2
    ∂ = RBMs.∂free_energy(layer)
    @test ∂.θ ≈ only(gs).θ ≈ -μ
    @test ∂.γ ≈ only(gs).γ ≈ μ2/2
    @test RBMs.grad2mean(layer, ∂) ≈ transfer_mean(layer)
    @test RBMs.grad2var(layer, ∂) ≈ transfer_var(layer)
end

@testset "pReLU / xReLU / dReLU convert" begin
    N = (10, 7)
    B = 13
    x = randn(N..., B)

    drelu = dReLU(randn(N...), randn(N...), rand(N...), rand(N...))
    prelu = @inferred pReLU(drelu)
    xrelu = @inferred xReLU(drelu)
    @test drelu.θp ≈ dReLU(prelu).θp ≈ dReLU(xrelu).θp
    @test drelu.θn ≈ dReLU(prelu).θn ≈ dReLU(xrelu).θn
    @test drelu.γp ≈ dReLU(prelu).γp
    @test drelu.γn ≈ dReLU(prelu).γn
    @test abs.(drelu.γp) ≈ dReLU(xrelu).γp
    @test abs.(drelu.γn) ≈ dReLU(xrelu).γn
    @test energies(drelu, x) ≈ energies(prelu, x) ≈ energies(xrelu, x)
    @test free_energies(drelu) ≈ free_energies(prelu) ≈ free_energies(xrelu)
    @test transfer_mode(drelu) ≈ transfer_mode(prelu) ≈ transfer_mode(xrelu)
    @test transfer_mean(drelu) ≈ transfer_mean(prelu) ≈ transfer_mean(xrelu)
    @test transfer_mean_abs(drelu) ≈ transfer_mean_abs(prelu) ≈ transfer_mean_abs(xrelu)
    @test transfer_var(drelu) ≈ transfer_var(prelu) ≈ transfer_var(xrelu)

    prelu = pReLU(randn(N...), rand(N...), randn(N...), 2rand(N...) .- 1)
    drelu = @inferred dReLU(prelu)
    xrelu = @inferred xReLU(prelu)
    @test prelu.θ ≈ pReLU(drelu).θ ≈ pReLU(xrelu).θ
    @test prelu.γ ≈ pReLU(drelu).γ ≈ pReLU(xrelu).γ
    @test prelu.Δ ≈ pReLU(drelu).Δ ≈ pReLU(xrelu).Δ
    @test prelu.η ≈ pReLU(drelu).η ≈ pReLU(xrelu).η
    @test energies(drelu, x) ≈ energies(prelu, x) ≈ energies(xrelu, x)
    @test free_energies(drelu) ≈ free_energies(prelu) ≈ free_energies(xrelu)
    @test transfer_mode(drelu) ≈ transfer_mode(prelu) ≈ transfer_mode(xrelu)
    @test transfer_mean(drelu) ≈ transfer_mean(prelu) ≈ transfer_mean(xrelu)
    @test transfer_mean_abs(drelu) ≈ transfer_mean_abs(prelu) ≈ transfer_mean_abs(xrelu)
    @test transfer_var(drelu) ≈ transfer_var(prelu) ≈ transfer_var(xrelu)

    xrelu = xReLU(randn(N...), rand(N...), randn(N...), randn(N...))
    drelu = @inferred dReLU(xrelu)
    prelu = @inferred pReLU(xrelu)
    @test xrelu.θ ≈ xReLU(drelu).θ ≈ xReLU(prelu).θ
    @test xrelu.Δ ≈ xReLU(drelu).Δ ≈ xReLU(prelu).Δ
    @test xrelu.ξ ≈ xReLU(drelu).ξ ≈ xReLU(prelu).ξ
    @test xrelu.γ ≈ xReLU(prelu).γ
    @test abs.(xrelu.γ) ≈ xReLU(drelu).γ
    @test energies(drelu, x) ≈ energies(prelu, x) ≈ energies(xrelu, x)
    @test free_energies(drelu) ≈ free_energies(prelu) ≈ free_energies(xrelu)
    @test transfer_mode(drelu) ≈ transfer_mode(prelu) ≈ transfer_mode(xrelu)
    @test transfer_mean(drelu) ≈ transfer_mean(prelu) ≈ transfer_mean(xrelu)
    @test transfer_mean_abs(drelu) ≈ transfer_mean_abs(prelu) ≈ transfer_mean_abs(xrelu)
    @test transfer_var(drelu) ≈ transfer_var(prelu) ≈ transfer_var(xrelu)

    gauss = Gaussian(randn(N...), rand(N...))
    drelu = @inferred dReLU(gauss)
    prelu = @inferred pReLU(gauss)
    xrelu = @inferred xReLU(gauss)
    @test (
        energies(gauss, x) ≈ energies(drelu, x) ≈
        energies(prelu, x) ≈ energies(xrelu, x)
    )
    @test (
        free_energies(gauss) ≈ free_energies(drelu) ≈
        free_energies(prelu) ≈ free_energies(xrelu)
    )
    @test (
        transfer_mode(gauss) ≈ transfer_mode(drelu) ≈
        transfer_mode(prelu) ≈ transfer_mode(xrelu)
    )
    @test (
        transfer_mean(gauss) ≈ transfer_mean(drelu) ≈
        transfer_mean(prelu) ≈ transfer_mean(xrelu)
    )
    @test (
        transfer_mean_abs(gauss) ≈ transfer_mean_abs(drelu) ≈
        transfer_mean_abs(prelu) ≈ transfer_mean_abs(xrelu)
    )
    @test (
        transfer_var(gauss) ≈ transfer_var(drelu) ≈
        transfer_var(prelu) ≈ transfer_var(xrelu)
    )

    drelu = dReLU(randn(1), [0.0], rand(1), [Inf])
    relu = ReLU(drelu.θp, drelu.γp)
    x = rand(1, 100)
    @test energies(relu, x) ≈ energies(drelu, x)
    @test free_energies(relu) ≈ free_energies(drelu)
    @test transfer_mode(relu) ≈ transfer_mode(drelu)
    #@test transfer_mean(relu) ≈ transfer_mean(drelu)
    #@test transfer_mean_abs(relu)  ≈ transfer_mean_abs(drelu)
    #@test transfer_var(relu) ≈ transfer_var(drelu)
end

@testset "dReLU" begin
    N = (3, 5)
    B = 13
    x = randn(N..., B)

    # bound γ away from zero to avoid issues with quadgk
    layer = dReLU(
        randn(N...), randn(N...), rand(N...) .+ 1, rand(N...) .+ 1
    )

    Ep = energy(ReLU( layer.θp, layer.γp), max.( x, 0))
    En = energy(ReLU(-layer.θn, layer.γn), max.(-x, 0))
    @test energy(layer, x) ≈ Ep + En
    @test iszero(energy(layer, zero(x)))

    function quad_free(θp::Real, θn::Real, γp::Real, γn::Real)
        Z, ϵ = quadgk(h -> exp(-RBMs.drelu_energy(θp, θn, γp, γn, h)), -Inf, Inf)
        return -log(Z)
    end
    @test free_energies(layer) ≈ quad_free.(layer.θp, layer.θn, abs.(layer.γp), abs.(layer.γn))

    gs = Zygote.gradient(layer) do layer
        sum(free_energies(layer))
    end
    ∂ = RBMs.∂free_energy(layer)
    @test ∂.θp ≈ only(gs).θp
    @test ∂.θn ≈ only(gs).θn
    @test ∂.γp ≈ only(gs).γp
    @test ∂.γn ≈ only(gs).γn
    μ = transfer_mean(layer)
    ν = transfer_var(layer)
    μ2 = @. ν + μ^2
    @test ∂.θp + ∂.θn ≈ -μ
    @test ∂.γp + ∂.γn ≈ μ2/2
    @test RBMs.grad2mean(layer, ∂) ≈ transfer_mean(layer)
    @test RBMs.grad2var(layer, ∂) ≈ transfer_var(layer)

    # check law of total variance
    inputs = randn(size(layer)..., 1000)
    ∂ = RBMs.∂free_energy(layer, inputs)
    h_ave = transfer_mean(layer, inputs)
    h_var = transfer_var(layer, inputs)
    μ = batchmean(layer, h_ave)
    ν_int = batchmean(layer, h_var)
    ν_ext = batchvar(layer, h_ave; mean = μ)
    ν = ν_int + ν_ext # law of total variance
    @test RBMs.grad2mean(layer, ∂) ≈ μ
    @test RBMs.grad2var(layer, ∂) ≈ ν
end

@testset "pReLU" begin
    N = (3, 5, 7)
    layer = pReLU(randn(N...), rand(N...), randn(N...), 2rand(N...) .- 1)
    gs = Zygote.gradient(layer) do layer
        sum(free_energies(layer))
    end
    ∂ = RBMs.∂free_energy(layer)
    @test ∂.θ ≈ only(gs).θ ≈ -transfer_mean(layer)
    @test ∂.γ ≈ only(gs).γ
    @test ∂.Δ ≈ only(gs).Δ
    @test ∂.η ≈ only(gs).η
    @test RBMs.grad2mean(layer, ∂) ≈ transfer_mean(layer)
    @test RBMs.grad2var(layer, ∂) ≈ transfer_var(layer)
end

@testset "xReLU" begin
    N = (3, 5, 7)
    layer = xReLU(randn(N...), rand(N...), randn(N...), randn(N...))
    gs = Zygote.gradient(layer) do layer
        sum(free_energies(layer))
    end
    ∂ = RBMs.∂free_energy(layer)
    @test ∂.θ ≈ only(gs).θ ≈ -transfer_mean(layer)
    @test ∂.γ ≈ only(gs).γ
    @test ∂.Δ ≈ only(gs).Δ
    @test ∂.ξ ≈ only(gs).ξ
    @test RBMs.grad2mean(layer, ∂) ≈ transfer_mean(layer)
    @test RBMs.grad2var(layer, ∂) ≈ transfer_var(layer)
end
