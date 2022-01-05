include("tests_init.jl")

Random.seed!(1)

_layers = (
    RBMs.Binary,
    RBMs.Spin,
    RBMs.Potts,
    RBMs.Gaussian,
    RBMs.ReLU,
    RBMs.dReLU,
    RBMs.pReLU,
    RBMs.xReLU
)

function random_layer(
    ::Type{T}, N::Int...
) where {T <: Union{RBMs.Binary, RBMs.Spin, RBMs.Potts}}
    return T(randn(N...))
end

function random_layer(::Type{T}, N::Int...) where {T <: Union{RBMs.Gaussian, RBMs.ReLU}}
    return T(randn(N...), rand(N...))
end

function random_layer(::Type{RBMs.dReLU}, N::Int...)
    return RBMs.dReLU(randn(N...), randn(N...), rand(N...), rand(N...))
end

function random_layer(::Type{RBMs.xReLU}, N::Int...)
    return RBMs.xReLU(randn(N...), rand(N...), randn(N...), randn(N...))
end

function random_layer(::Type{RBMs.pReLU}, N::Int...)
    return RBMs.pReLU(randn(N...), rand(N...), randn(N...), 2rand(N...) .- 1)
end

@testset "testing $Layer" for Layer in _layers
    N = (3, 2)
    B = 23
    layer = random_layer(Layer, N...)

    @test (@inferred size(layer)) == N
    @test (@inferred length(layer)) == prod(N)
    @test (@inferred ndims(layer)) == length(N)

    x = bitrand(N..., B)
    inputs = randn(N..., B)
    β = rand()

    @test size(RBMs.energy(layer, x)) == (B,)
    @test size(RBMs.free_energy(layer, inputs; β=1)) == (B,)
    @test size(RBMs.transfer_sample(layer, inputs; β=1)) == size(inputs)
    @test size(RBMs.transfer_sample(layer, 0; β=1)) == size(RBMs.transfer_sample(layer)) == size(layer)
    @test RBMs.free_energy(layer, inputs; β=1) ≈ RBMs.free_energy(layer, inputs)
    @test RBMs.free_energies(layer, 0; β=1) ≈ RBMs.free_energies(layer)
    @test RBMs.free_energy(layer) isa Real
    @inferred RBMs.energies(layer, x)
    @inferred RBMs.energy(layer, x)
    @inferred RBMs.free_energies(layer)
    @inferred RBMs.free_energy(layer)
    @inferred RBMs.transfer_sample(layer)
    @inferred RBMs.transfer_mean(layer)
    @inferred RBMs.transfer_var(layer)

    @test RBMs.energy(layer, x) ≈ vec(sum(RBMs.energies(layer, x); dims=1:ndims(layer)))
    @test RBMs.free_energy(layer, inputs; β) ≈ vec(sum(RBMs.free_energies(layer, inputs; β); dims=1:ndims(layer)))

    μ = RBMs.transfer_mean(layer, inputs)
    @test only(Zygote.gradient(j -> sum(RBMs.free_energies(layer, j)), inputs)) ≈ -μ

    samples = RBMs.transfer_sample(layer, zeros(size(layer)..., 10^6))
    @test RBMs.transfer_mean(layer) ≈ RBMs.mean_(samples; dims=ndims(samples)) rtol=0.1
    @test RBMs.transfer_var(layer) ≈ RBMs.var_(samples;  dims=ndims(samples)) rtol=0.1
    @test RBMs.transfer_std(layer) ≈ sqrt.(RBMs.transfer_var(layer))
    @test RBMs.transfer_mean_abs(layer) ≈ RBMs.mean_(abs.(samples); dims=ndims(samples)) rtol=0.1
    @test all(RBMs.energy(layer, RBMs.transfer_mode(layer)) .≤ RBMs.energy(layer, samples))

    ∂F = RBMs.∂free_energy(layer)
    ∂E = RBMs.∂energy(layer, samples)
    @test length(∂F) == length(∂E)
    @test typeof(∂F) == typeof(∂E)
    for (∂f, ∂e) in zip(∂F, ∂E)
        @test ∂f ≈ ∂e rtol=0.1
    end

    gs = Zygote.gradient(Flux.params(layer)) do
        sum(RBMs.energies(layer, samples)) / size(samples)[end]
    end
    for (ω, ∂ω) in pairs(∂E)
        @test ∂ω ≈ gs[getproperty(layer, ω)]
    end

    ∂F = RBMs.∂free_energy(layer)
    gs = Zygote.gradient(Flux.params(layer)) do
        sum(RBMs.free_energies(layer))
    end
    for (ω, ∂ω) in pairs(∂F)
        @test ∂ω ≈ gs[getproperty(layer, ω)]
    end
end

@testset "discrete layers ($Layer)" for Layer in (RBMs.Binary, RBMs.Spin, RBMs.Potts)
    N = (3, 4, 5)
    B = 13
    layer = Layer(randn(N...))
    x = bitrand(N..., B)
    @test RBMs.energies(layer, x) ≈ -layer.θ .* x

    ps = Flux.params(layer)
    gs = Zygote.gradient(ps) do
        sum(RBMs.free_energies(layer))
    end
    @test RBMs.∂free_energy(layer).θ ≈ gs[layer.θ] ≈ -RBMs.transfer_mean(layer)
end

@testset "Binary" begin
    @testset "binary_rand" begin
        binary_rand(θ, u) = u * (1 + exp(-θ)) < 1

        for θ in -5:5, u in 0.0:0.1:1.0
            @test binary_rand(θ, u) == @inferred RBMs.binary_rand(θ, u)
        end

        θ = randn(1000)
        u = rand(1000)
        @test RBMs.binary_rand.(θ, u) == binary_rand.(θ, u)
    end

    @testset "binary_var" begin
        θ = randn(1000)
        @test RBMs.binary_var.(θ) ≈ @. LogExpFunctions.logistic(θ) * LogExpFunctions.logistic(-θ)
    end

    layer = RBMs.Binary(randn(7, 4, 5))
    @test RBMs.free_energies(layer) ≈ -log.(sum(exp.(layer.θ .* h) for h in 0:1))
    @test sort(unique(RBMs.transfer_sample(layer))) == [0, 1]
end

@testset "Spin" begin
    layer = RBMs.Spin(randn(7, 4, 5))
    @test RBMs.free_energies(layer) ≈ -log.(sum(exp.(layer.θ .* h) for h in (-1, 1)))
    @test sort(unique(RBMs.transfer_sample(layer))) == [-1, 1]
end

@testset "Potts" begin
    q = 3
    N = (4, 5)
    layer = RBMs.Potts(randn(q, N...))
    @test RBMs.free_energies(layer) ≈ -log.(sum(exp.(layer.θ[h:h,:,:,:]) for h in 1:q))
    @test all(sum(RBMs.transfer_mean(layer); dims=1) .≈ 1)
    # samples are proper one-hot
    @test sort(unique(RBMs.transfer_sample(layer))) == [0, 1]
    @test all(sum(RBMs.transfer_sample(layer); dims=1) .== 1)
end

@testset "Gaussian" begin
    N = (3, 4, 5)
    B = 7

    # bound γ away from zero to avoid numerical issues with QuadGK
    layer = RBMs.Gaussian(randn(N...), rand(N...) .+ 0.5)

    x = randn(size(layer)..., B)
    @test RBMs.energies(layer, x) ≈ @. abs(layer.γ) * x^2 / 2 - layer.θ * x

    function quad_free(θ::Real, γ::Real)
        Z, ϵ = QuadGK.quadgk(h -> exp(-RBMs.gauss_energy(θ, γ, h)), -Inf,  Inf)
        return -log(Z)
    end

    @test RBMs.free_energies(layer) ≈ quad_free.(layer.θ, layer.γ) rtol=1e-6

    gs = Zygote.gradient(Flux.params(layer)) do
        sum(RBMs.free_energies(layer))
    end
    μ = RBMs.transfer_mean(layer)
    ν = RBMs.transfer_var(layer)
    μ2 = @. ν + μ^2
    ∂ = RBMs.∂free_energy(layer)
    @test ∂.θ ≈ gs[layer.θ] ≈ -μ
    @test ∂.γ ≈ gs[layer.γ] ≈ μ2/2
end

@testset "ReLU" begin
    N = (3, 4, 5)
    B = 7

    # bound γ away from zero to avoid numerical issues with QuadGK
    layer = RBMs.ReLU(randn(N...), rand(N...) .+ 0.5)

    x = abs.(randn(size(layer)..., B))
    @test RBMs.energies(layer, x) ≈ RBMs.energies(RBMs.Gaussian(layer.θ, layer.γ), x)

    function quad_free(θ::Real, γ::Real)
        Z, ϵ = QuadGK.quadgk(h -> exp(-RBMs.relu_energy(θ, γ, h)), 0,  Inf)
        return -log(Z)
    end
    @test RBMs.free_energies(layer) ≈ @. quad_free(layer.θ, layer.γ)

    gs = Zygote.gradient(Flux.params(layer)) do
        sum(RBMs.free_energies(layer))
    end
    μ = RBMs.transfer_mean(layer)
    ν = RBMs.transfer_var(layer)
    μ2 = @. ν + μ^2
    ∂ = RBMs.∂free_energy(layer)
    @test ∂.θ ≈ gs[layer.θ] ≈ -μ
    @test ∂.γ ≈ gs[layer.γ] ≈ μ2/2
end

@testset "pReLU / xReLU / dReLU convert" begin
    N = (10, 7)
    B = 13
    x = randn(N..., B)

    drelu = RBMs.dReLU(randn(N...), randn(N...), rand(N...), rand(N...))
    prelu = @inferred RBMs.pReLU(drelu)
    xrelu = @inferred RBMs.xReLU(drelu)
    @test drelu.θp ≈ RBMs.dReLU(prelu).θp ≈ RBMs.dReLU(xrelu).θp
    @test drelu.θn ≈ RBMs.dReLU(prelu).θn ≈ RBMs.dReLU(xrelu).θn
    @test drelu.γp ≈ RBMs.dReLU(prelu).γp
    @test drelu.γn ≈ RBMs.dReLU(prelu).γn
    @test abs.(drelu.γp) ≈ RBMs.dReLU(xrelu).γp
    @test abs.(drelu.γn) ≈ RBMs.dReLU(xrelu).γn
    @test RBMs.energies(drelu, x) ≈ RBMs.energies(prelu, x) ≈ RBMs.energies(xrelu, x)
    @test RBMs.free_energies(drelu) ≈ RBMs.free_energies(prelu) ≈ RBMs.free_energies(xrelu)
    @test RBMs.transfer_mode(drelu) ≈ RBMs.transfer_mode(prelu) ≈ RBMs.transfer_mode(xrelu)
    @test RBMs.transfer_mean(drelu) ≈ RBMs.transfer_mean(prelu) ≈ RBMs.transfer_mean(xrelu)
    @test RBMs.transfer_mean_abs(drelu) ≈ RBMs.transfer_mean_abs(prelu) ≈ RBMs.transfer_mean_abs(xrelu)
    @test RBMs.transfer_var(drelu) ≈ RBMs.transfer_var(prelu) ≈ RBMs.transfer_var(xrelu)

    prelu = RBMs.pReLU(randn(N...), rand(N...), randn(N...), 2rand(N...) .- 1)
    drelu = @inferred RBMs.dReLU(prelu)
    xrelu = @inferred RBMs.xReLU(prelu)
    @test prelu.θ ≈ RBMs.pReLU(drelu).θ ≈ RBMs.pReLU(xrelu).θ
    @test prelu.γ ≈ RBMs.pReLU(drelu).γ ≈ RBMs.pReLU(xrelu).γ
    @test prelu.Δ ≈ RBMs.pReLU(drelu).Δ ≈ RBMs.pReLU(xrelu).Δ
    @test prelu.η ≈ RBMs.pReLU(drelu).η ≈ RBMs.pReLU(xrelu).η
    @test RBMs.energies(drelu, x) ≈ RBMs.energies(prelu, x) ≈ RBMs.energies(xrelu, x)
    @test RBMs.free_energies(drelu) ≈ RBMs.free_energies(prelu) ≈ RBMs.free_energies(xrelu)
    @test RBMs.transfer_mode(drelu) ≈ RBMs.transfer_mode(prelu) ≈ RBMs.transfer_mode(xrelu)
    @test RBMs.transfer_mean(drelu) ≈ RBMs.transfer_mean(prelu) ≈ RBMs.transfer_mean(xrelu)
    @test RBMs.transfer_mean_abs(drelu) ≈ RBMs.transfer_mean_abs(prelu) ≈ RBMs.transfer_mean_abs(xrelu)
    @test RBMs.transfer_var(drelu) ≈ RBMs.transfer_var(prelu) ≈ RBMs.transfer_var(xrelu)

    xrelu = RBMs.xReLU(randn(N...), rand(N...), randn(N...), randn(N...))
    drelu = @inferred RBMs.dReLU(xrelu)
    prelu = @inferred RBMs.pReLU(xrelu)
    @test xrelu.θ ≈ RBMs.xReLU(drelu).θ ≈ RBMs.xReLU(prelu).θ
    @test xrelu.Δ ≈ RBMs.xReLU(drelu).Δ ≈ RBMs.xReLU(prelu).Δ
    @test xrelu.ξ ≈ RBMs.xReLU(drelu).ξ ≈ RBMs.xReLU(prelu).ξ
    @test xrelu.γ ≈ RBMs.xReLU(prelu).γ
    @test abs.(xrelu.γ) ≈ RBMs.xReLU(drelu).γ
    @test RBMs.energies(drelu, x) ≈ RBMs.energies(prelu, x) ≈ RBMs.energies(xrelu, x)
    @test RBMs.free_energies(drelu) ≈ RBMs.free_energies(prelu) ≈ RBMs.free_energies(xrelu)
    @test RBMs.transfer_mode(drelu) ≈ RBMs.transfer_mode(prelu) ≈ RBMs.transfer_mode(xrelu)
    @test RBMs.transfer_mean(drelu) ≈ RBMs.transfer_mean(prelu) ≈ RBMs.transfer_mean(xrelu)
    @test RBMs.transfer_mean_abs(drelu) ≈ RBMs.transfer_mean_abs(prelu) ≈ RBMs.transfer_mean_abs(xrelu)
    @test RBMs.transfer_var(drelu) ≈ RBMs.transfer_var(prelu) ≈ RBMs.transfer_var(xrelu)

    gauss = RBMs.Gaussian(randn(N...), rand(N...))
    drelu = @inferred RBMs.dReLU(gauss)
    prelu = @inferred RBMs.pReLU(gauss)
    xrelu = @inferred RBMs.xReLU(gauss)
    @test (
        RBMs.energies(gauss, x) ≈ RBMs.energies(drelu, x) ≈
        RBMs.energies(prelu, x) ≈ RBMs.energies(xrelu, x)
    )
    @test (
        RBMs.free_energies(gauss) ≈ RBMs.free_energies(drelu) ≈
        RBMs.free_energies(prelu) ≈ RBMs.free_energies(xrelu)
    )
    @test (
        RBMs.transfer_mode(gauss) ≈ RBMs.transfer_mode(drelu) ≈
        RBMs.transfer_mode(prelu) ≈ RBMs.transfer_mode(xrelu)
    )
    @test (
        RBMs.transfer_mean(gauss) ≈ RBMs.transfer_mean(drelu) ≈
        RBMs.transfer_mean(prelu) ≈ RBMs.transfer_mean(xrelu)
    )
    @test (
        RBMs.transfer_mean_abs(gauss) ≈ RBMs.transfer_mean_abs(drelu) ≈
        RBMs.transfer_mean_abs(prelu) ≈ RBMs.transfer_mean_abs(xrelu)
    )
    @test (
        RBMs.transfer_var(gauss) ≈ RBMs.transfer_var(drelu) ≈
        RBMs.transfer_var(prelu) ≈ RBMs.transfer_var(xrelu)
    )

    # relu  = RBMs.ReLU(randn(N...), rand(N...))
    # drelu = @inferred RBMs.dReLU(relu)
    # @test RBMs.energies(relu, x) ≈ RBMs.energies(drelu, x)
    # @test RBMs.free_energies(relu) ≈ RBMs.free_energies(drelu)
    # @test RBMs.transfer_mode(relu) ≈ RBMs.transfer_mode(drelu)
    # @test RBMs.transfer_mean(relu) ≈ RBMs.transfer_mean(drelu)
    # @test RBMs.transfer_mean_abs(relu)  ≈ RBMs.transfer_mean_abs(drelu)
    # @test RBMs.transfer_var(relu) ≈ RBMs.transfer_var(drelu)
end

@testset "dReLU" begin
    N = (3, 5)
    B = 13
    x = randn(N..., B)

    # bound γ away from zero to avoid issues with QuadGK
    layer = RBMs.dReLU(randn(N...), randn(N...), rand(N...) .+ 0.5, rand(N...) .+ 0.5)

    Ep = RBMs.energy(RBMs.ReLU( layer.θp, layer.γp), max.( x, 0))
    En = RBMs.energy(RBMs.ReLU(-layer.θn, layer.γn), max.(-x, 0))
    @test RBMs.energy(layer, x) ≈ Ep + En
    @test iszero(RBMs.energy(layer, zero(x)))

    function quad_free(θp::Real, θn::Real, γp::Real, γn::Real)
        Z, ϵ = QuadGK.quadgk(h -> exp(-RBMs.drelu_energy(θp, θn, γp, γn, h)), -Inf, Inf)
        return -log(Z)
    end

    @test RBMs.free_energies(layer) ≈ quad_free.(layer.θp, layer.θn, abs.(layer.γp), abs.(layer.γn))

    gs = Zygote.gradient(Flux.params(layer)) do
        sum(RBMs.free_energies(layer))
    end
    ∂ = RBMs.∂free_energy(layer)
    @test ∂.θp ≈ gs[layer.θp]
    @test ∂.θn ≈ gs[layer.θn]
    @test ∂.γp ≈ gs[layer.γp]
    @test ∂.γn ≈ gs[layer.γn]
end

@testset "pReLU" begin
    N = (3, 5)
    layer = RBMs.pReLU(randn(N...), rand(N...), randn(N...), 2rand(N...) .- 1)
    gs = Zygote.gradient(Flux.params(layer)) do
        sum(RBMs.free_energies(layer))
    end
    ∂ = RBMs.∂free_energy(layer)
    @test ∂.θ ≈ gs[layer.θ]
    @test ∂.γ ≈ gs[layer.γ]
    @test ∂.Δ ≈ gs[layer.Δ]
    @test ∂.η ≈ gs[layer.η]
end

@testset "xReLU" begin
    N = (3, 5)
    layer = RBMs.xReLU(randn(N...), rand(N...), randn(N...), randn(N...))
    gs = Zygote.gradient(Flux.params(layer)) do
        sum(RBMs.free_energies(layer))
    end
    ∂ = RBMs.∂free_energy(layer)
    @test ∂.θ ≈ gs[layer.θ]
    @test ∂.γ ≈ gs[layer.γ]
    @test ∂.Δ ≈ gs[layer.Δ]
    @test ∂.ξ ≈ gs[layer.ξ]
end
