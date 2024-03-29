import Random
import Zygote
using Base: tail
using Test: @test, @testset, @inferred
using Statistics: mean, var, cov
using Random: bitrand, rand!, randn!
using LogExpFunctions: logistic
using EllipsisNotation: (..)
using QuadGK: quadgk
using RestrictedBoltzmannMachines: RBM, Binary, Spin, Potts, Gaussian, ReLU, dReLU, xReLU, pReLU,
    flatten, batch_size, batchmean, batchvar, batchcov, grad2ave, drelu_energy,
    mean_from_inputs, var_from_inputs, meanvar_from_inputs, batchdims, gauss_energy, relu_energy,
    std_from_inputs, mean_abs_from_inputs, sample_from_inputs, mode_from_inputs,
    energy, cgf, free_energy, cgfs, energies, ∂cgf, vstack, ∂energy, ∂free_energy, binary_rand,
    total_meanvar_from_inputs, total_mean_from_inputs, total_var_from_inputs, sample_v_from_v

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

@testset "testing $Layer" for Layer in _layers
    sz = (3,2)
    layer = Layer(sz)
    randn!(layer.par)

    if layer isa pReLU
        layer.η .= layer.η ./ (1 .+ abs.(layer.η)) # make sure -1 ≤ η ≤ 1
    end

    @test (@inferred size(layer)) == sz
    for (d, n) in enumerate(sz)
        @test (@inferred size(layer, d)) == n
    end

    @test (@inferred length(layer)) == prod(sz)
    @test (@inferred ndims(layer)) == length(sz)
    @test (@inferred batch_size(layer, rand(sz...))) == ()
    @test (@inferred energy(layer, rand(sz...))) isa Number
    @test (@inferred cgf(layer)) isa Number
    @test (@inferred cgf(layer, rand(sz...))) isa Number
    @test size(@inferred mean_from_inputs(layer)) == size(layer)
    @test size(@inferred var_from_inputs(layer)) == size(layer)
    @test size(@inferred sample_from_inputs(layer)) == size(layer)
    @test cgfs(layer, 0) ≈ cgfs(layer)
    @test std_from_inputs(layer) ≈ sqrt.(var_from_inputs(layer))

    if layer isa Potts
        @test size(@inferred cgfs(layer)) == (1, tail(size(layer))...)
    else
        @test size(@inferred cgfs(layer)) == size(layer)
    end

    @test size(@inferred sample_from_inputs(layer, 0)) == size(layer)

    for B in ((), (2,), (1,2))
        x = rand(sz..., B...)
        @test (@inferred batch_size(layer, x)) == (B...,)
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

        if layer isa Potts
            @test size(@inferred cgfs(layer, x)) == (1, tail(size(x))...)
        else
            @test size(@inferred cgfs(layer, x)) == size(x)
        end

        if B == ()
            @test @inferred(energy(layer, x)) ≈ sum(energies(layer, x))
            @test @inferred(cgf(layer, x)) ≈ sum(cgfs(layer, x))
            @test @inferred(batchmean(layer, x)) ≈ x
            @test @inferred(batchvar(layer, x)) == zeros(sz)
            @test @inferred(batchcov(layer, x)) == zeros(sz..., sz...)
        else
            @test @inferred(energy(layer, x)) ≈ reshape(sum(energies(layer, x); dims=1:ndims(layer)), B)
            @test @inferred(cgf(layer, x)) ≈ reshape(sum(cgfs(layer, x); dims=1:ndims(layer)), B)
            @test @inferred(batchmean(layer, x)) ≈ reshape(mean(x; dims=(ndims(layer) + 1):ndims(x)), sz)
            @test @inferred(batchvar(layer, x)) ≈ reshape(var(x; dims=(ndims(layer) + 1):ndims(x), corrected=false), sz)
            @test @inferred(batchcov(layer, x)) ≈ reshape(cov(flatten(layer, x); dims=2, corrected=false), sz..., sz...)
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

@testset "discrete layers ($Layer)" for Layer in (Binary, Spin, Potts)
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

@testset "Binary" begin
    @testset "binary_rand" begin
        for θ in -5:5, u in 0.0:0.1:1.0
            @test (@. u * (1 + exp(-θ)) < 1) == @inferred binary_rand(θ, u)
        end
        θ = randn(1000)
        u = rand(1000)
        @test binary_rand.(θ, u) == (@. u * (1 + exp(-θ)) < 1)
    end

    layer = Binary(; θ = randn(7, 4, 5))
    var_from_inputs(layer) ≈ @. logistic(layer.θ) * logistic(-layer.θ)
    @test cgfs(layer) ≈ log.(sum(exp.(layer.θ .* h) for h in 0:1))
    @test sort(unique(sample_from_inputs(layer))) == [0, 1]

    gs = Zygote.gradient(layer) do layer
        sum(cgfs(layer))
    end
    ∂ = ∂cgf(layer)
    @test ∂ ≈ only(gs).par ≈ vstack((mean_from_inputs(layer),))
    @test grad2ave(layer, ∂) ≈ mean_from_inputs(layer)
end

@testset "Spin" begin
    layer = Spin(; θ = randn(7, 4, 5))
    @test cgfs(layer) ≈ log.(sum(exp.(layer.θ .* h) for h in (-1, 1)))
    @test sort(unique(sample_from_inputs(layer))) == [-1, 1]

    gs = Zygote.gradient(layer) do layer
        sum(cgfs(layer))
    end
    ∂ = ∂cgf(layer)
    @test ∂ ≈ only(gs).par ≈ vstack((mean_from_inputs(layer),))
    @test grad2ave(layer, ∂) ≈ mean_from_inputs(layer)
end

@testset "Potts" begin
    q = 3
    N = (4, 5)
    layer = Potts(; θ = randn(q, N...))
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

@testset "Gaussian" begin
    N = (3, 4, 5)
    B = 7

    # bound γ away from zero to avoid numerical issues with quadgk, but allow negative
    layer = Gaussian(; θ = randn(N...), γ = (rand(N...) .+ 1) .* (rand(Bool, N...) .- 0.5))

    x = randn(size(layer)..., B)
    @test energies(layer, x) ≈ @. abs(layer.γ) * x^2 / 2 - layer.θ * x

    function quad_cgf(θ::Real, γ::Real)
        Z, ϵ = quadgk(h -> exp(-gauss_energy(θ, γ, h)), -Inf,  Inf)
        return log(Z)
    end

    @test cgfs(layer) ≈ quad_cgf.(layer.θ, layer.γ) rtol=1e-6

    gs = Zygote.gradient(layer) do layer
        sum(cgfs(layer))
    end
    μ = mean_from_inputs(layer)
    ν = var_from_inputs(layer)
    μ2 = @. ν + μ^2
    ∂ = ∂cgf(layer)
    @test ∂ ≈ only(gs).par
    @test ∂[1, ..] ≈ μ
    @test ∂[2, ..] ≈ -sign.(layer.γ) .* μ2/2
    @test grad2ave(layer, ∂) ≈ mean_from_inputs(layer)
end

@testset "ReLU" begin
    N = (3, 4, 5)
    B = 7

    # bound γ away from zero to avoid numerical issues with quadgk, but allow negative
    layer = ReLU(; θ = randn(N...), γ = (rand(N...) .+ 1) .* (rand(Bool, N...) .- 0.5))

    x = abs.(randn(size(layer)..., B))
    @test energies(layer, x) ≈ energies(Gaussian(; layer.θ, layer.γ), x)

    function quad_cgf(θ::Real, γ::Real)
        Z, ϵ = quadgk(h -> exp(-relu_energy(θ, γ, h)), 0,  Inf)
        return log(Z)
    end
    @test cgfs(layer) ≈ @. quad_cgf(layer.θ, layer.γ)

    gs = Zygote.gradient(layer) do layer
        sum(cgfs(layer))
    end
    μ = mean_from_inputs(layer)
    ν = var_from_inputs(layer)
    μ2 = @. ν + μ^2
    ∂ = ∂cgf(layer)

    @test ∂ ≈ only(gs).par
    @test ∂[1, ..] ≈ μ
    @test ∂[2, ..] ≈ -sign.(layer.γ) .* μ2/2
    @test grad2ave(layer, ∂) ≈ mean_from_inputs(layer)
end

@testset "pReLU / xReLU / dReLU convert" begin
    N = (10, 7)
    B = 13
    x = randn(N..., B)

    drelu = dReLU(; θp = randn(N...), θn = randn(N...), γp = randn(N...), γn = randn(N...))
    prelu = @inferred pReLU(drelu)
    xrelu = @inferred xReLU(drelu)
    @test drelu.θp ≈ dReLU(prelu).θp ≈ dReLU(xrelu).θp
    @test drelu.θn ≈ dReLU(prelu).θn ≈ dReLU(xrelu).θn
    @test abs.(drelu.γp) ≈ abs.(dReLU(prelu).γp)
    @test abs.(drelu.γn) ≈ abs.(dReLU(prelu).γn)
    @test abs.(drelu.γp) ≈ abs.(dReLU(xrelu).γp)
    @test abs.(drelu.γn) ≈ abs.(dReLU(xrelu).γn)
    @test energies(drelu, x) ≈ energies(prelu, x) ≈ energies(xrelu, x)
    @test cgfs(drelu) ≈ cgfs(prelu) ≈ cgfs(xrelu)
    @test mode_from_inputs(drelu) ≈ mode_from_inputs(prelu) ≈ mode_from_inputs(xrelu)
    @test mean_from_inputs(drelu) ≈ mean_from_inputs(prelu) ≈ mean_from_inputs(xrelu)
    @test mean_abs_from_inputs(drelu) ≈ mean_abs_from_inputs(prelu) ≈ mean_abs_from_inputs(xrelu)
    @test var_from_inputs(drelu) ≈ var_from_inputs(prelu) ≈ var_from_inputs(xrelu)

    prelu = pReLU(; θ = randn(N...), γ = randn(N...), Δ = randn(N...), η = 2rand(N...) .- 1)
    drelu = @inferred dReLU(prelu)
    xrelu = @inferred xReLU(prelu)
    @test prelu.θ ≈ pReLU(drelu).θ ≈ pReLU(xrelu).θ
    @test abs.(prelu.γ) ≈ abs.(pReLU(drelu).γ) ≈ abs.(pReLU(xrelu).γ)
    @test prelu.Δ ≈ pReLU(drelu).Δ ≈ pReLU(xrelu).Δ
    @test prelu.η ≈ pReLU(drelu).η ≈ pReLU(xrelu).η
    @test energies(drelu, x) ≈ energies(prelu, x) ≈ energies(xrelu, x)
    @test cgfs(drelu) ≈ cgfs(prelu) ≈ cgfs(xrelu)
    @test mode_from_inputs(drelu) ≈ mode_from_inputs(prelu) ≈ mode_from_inputs(xrelu)
    @test mean_from_inputs(drelu) ≈ mean_from_inputs(prelu) ≈ mean_from_inputs(xrelu)
    @test mean_abs_from_inputs(drelu) ≈ mean_abs_from_inputs(prelu) ≈ mean_abs_from_inputs(xrelu)
    @test var_from_inputs(drelu) ≈ var_from_inputs(prelu) ≈ var_from_inputs(xrelu)

    xrelu = xReLU(; θ = randn(N...), γ = randn(N...), Δ = randn(N...), ξ = randn(N...))
    drelu = @inferred dReLU(xrelu)
    prelu = @inferred pReLU(xrelu)
    @test xrelu.θ ≈ xReLU(drelu).θ ≈ xReLU(prelu).θ
    @test xrelu.Δ ≈ xReLU(drelu).Δ ≈ xReLU(prelu).Δ
    @test xrelu.ξ ≈ xReLU(drelu).ξ ≈ xReLU(prelu).ξ
    @test abs.(xrelu.γ) ≈ abs.(xReLU(prelu).γ)
    @test energies(drelu, x) ≈ energies(prelu, x) ≈ energies(xrelu, x)
    @test cgfs(drelu) ≈ cgfs(prelu) ≈ cgfs(xrelu)
    @test mode_from_inputs(drelu) ≈ mode_from_inputs(prelu) ≈ mode_from_inputs(xrelu)
    @test mean_from_inputs(drelu) ≈ mean_from_inputs(prelu) ≈ mean_from_inputs(xrelu)
    @test mean_abs_from_inputs(drelu) ≈ mean_abs_from_inputs(prelu) ≈ mean_abs_from_inputs(xrelu)
    @test var_from_inputs(drelu) ≈ var_from_inputs(prelu) ≈ var_from_inputs(xrelu)

    gauss = Gaussian(; θ = randn(N...), γ = randn(N...))
    drelu = @inferred dReLU(gauss)
    prelu = @inferred pReLU(gauss)
    xrelu = @inferred xReLU(gauss)
    @test (
        energies(gauss, x) ≈ energies(drelu, x) ≈
        energies(prelu, x) ≈ energies(xrelu, x)
    )
    @test (
        cgfs(gauss) ≈ cgfs(drelu) ≈
        cgfs(prelu) ≈ cgfs(xrelu)
    )
    @test (
        mode_from_inputs(gauss) ≈ mode_from_inputs(drelu) ≈
        mode_from_inputs(prelu) ≈ mode_from_inputs(xrelu)
    )
    @test (
        mean_from_inputs(gauss) ≈ mean_from_inputs(drelu) ≈
        mean_from_inputs(prelu) ≈ mean_from_inputs(xrelu)
    )
    @test (
        mean_abs_from_inputs(gauss) ≈ mean_abs_from_inputs(drelu) ≈
        mean_abs_from_inputs(prelu) ≈ mean_abs_from_inputs(xrelu)
    )
    @test (
        var_from_inputs(gauss) ≈ var_from_inputs(drelu) ≈
        var_from_inputs(prelu) ≈ var_from_inputs(xrelu)
    )

    drelu = dReLU(; θp = randn(1), θn = [0.0], γp = randn(1), γn = [Inf])
    relu = ReLU(; θ = drelu.θp, γ = drelu.γp)
    x = rand(1, 100)
    @test energies(relu, x) ≈ energies(drelu, x)
    @test cgfs(relu) ≈ cgfs(drelu)
    @test mode_from_inputs(relu) ≈ mode_from_inputs(drelu)
    #@test mean_from_inputs(relu) ≈ mean_from_inputs(drelu)
    #@test mean_abs_from_inputs(relu)  ≈ mean_abs_from_inputs(drelu)
    #@test var_from_inputs(relu) ≈ var_from_inputs(drelu)
end

@testset "dReLU" begin
    N = (3, 5)
    B = 13
    x = randn(N..., B)

    # bound γ away from zero to avoid issues with quadgk
    layer = dReLU(
        θp = randn(N...), θn = randn(N...),
        γp = (rand(N...) .+ 1) .* (rand(Bool, N...) .- 0.5),
        γn = (rand(N...) .+ 1) .* (rand(Bool, N...) .- 0.5)
    )

    Ep = energy(ReLU(; θ =  layer.θp, γ = layer.γp), max.( x, 0))
    En = energy(ReLU(; θ = -layer.θn, γ = layer.γn), max.(-x, 0))
    @test energy(layer, x) ≈ Ep + En
    @test iszero(energy(layer, zero(x)))

    function quad_cgf(θp::Real, θn::Real, γp::Real, γn::Real)
        Z, ϵ = quadgk(h -> exp(-drelu_energy(θp, θn, γp, γn, h)), -Inf, Inf)
        return log(Z)
    end
    @test cgfs(layer) ≈ quad_cgf.(layer.θp, layer.θn, layer.γp, layer.γn)

    gs = Zygote.gradient(layer) do layer
        sum(cgfs(layer))
    end
    ∂ = @inferred ∂cgf(layer)
    @test ∂ ≈ only(gs).par
    @test grad2ave(layer, ∂) ≈ mean_from_inputs(layer)

    # check law of total variance
    inputs = randn(size(layer)..., 1000)
    ∂ = ∂cgf(layer, inputs)
    h_ave = mean_from_inputs(layer, inputs)
    h_var = var_from_inputs(layer, inputs)
    μ = batchmean(layer, h_ave)
    ν_int = batchmean(layer, h_var)
    ν_ext = batchvar(layer, h_ave; mean = μ)
    ν = ν_int + ν_ext # law of total variance
    @test grad2ave(layer, ∂) ≈ μ
    μ1, ν1 = total_meanvar_from_inputs(layer, inputs)
    @test μ1 ≈ μ ≈ total_mean_from_inputs(layer, inputs)
    @test ν1 ≈ ν ≈ total_var_from_inputs(layer, inputs)
end

@testset "pReLU" begin
    N = (3, 5, 7)
    layer = pReLU(; θ = randn(N...), γ = randn(N...), Δ = randn(N...), η = 2rand(N...) .- 1)
    gs = Zygote.gradient(layer) do layer
        sum(cgfs(layer))
    end
    ∂ = ∂cgf(layer)
    @test ∂ ≈ only(gs).par
    @test grad2ave(layer, ∂) ≈ mean_from_inputs(layer)
end

@testset "xReLU" begin
    N = (3, 5, 7)
    layer = xReLU(; θ = randn(N...), γ = randn(N...), Δ = randn(N...), ξ = randn(N...))
    gs = Zygote.gradient(layer) do layer
        sum(cgfs(layer))
    end
    ∂ = ∂cgf(layer)
    @test ∂ ≈ only(gs).par
    @test grad2ave(layer, ∂) ≈ mean_from_inputs(layer)
end

@testset "grad2ave $Layer" for Layer in _layers
    layer = Layer((5,))
    rbm = RBM(layer, Binary(; θ = randn(3)), randn(5,3))
    v = sample_v_from_v(rbm, randn(5,100); steps=100)
    ∂ = ∂free_energy(rbm, v)
    @test (@inferred grad2ave(rbm.visible, -∂.visible)) ≈ dropdims(mean(v; dims=2); dims=2)
end
