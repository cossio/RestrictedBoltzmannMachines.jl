include("tests_init.jl")

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
    @test size(RBMs.cgf(layer, inputs, 1)) == (B,)
    @test size(RBMs.transfer_sample(layer, inputs, 1)) == size(inputs)
    @test size(RBMs.transfer_sample(layer, 0, 1)) == size(RBMs.transfer_sample(layer)) == size(layer)
    @test RBMs.cgf(layer, inputs, 1) ≈ RBMs.cgf(layer, inputs)
    @test RBMs.cgfs(layer, 0, 1) ≈ RBMs.cgfs(layer)
    @test RBMs.cgf(layer) isa Real
    @inferred RBMs.energies(layer, x)
    @inferred RBMs.energy(layer, x)
    @inferred RBMs.cgfs(layer)
    @inferred RBMs.cgf(layer)
    @inferred RBMs.transfer_sample(layer)
    @inferred RBMs.transfer_mean(layer)
    @inferred RBMs.transfer_var(layer)

    @test RBMs.energy(layer, x) ≈ vec(sum(RBMs.energies(layer, x); dims=1:ndims(layer)))
    @test RBMs.cgf(layer, inputs, β) ≈ vec(sum(RBMs.cgfs(layer, inputs, β); dims=1:ndims(layer)))

    μ = RBMs.transfer_mean(layer, inputs)
    @test only(Zygote.gradient(j -> sum(RBMs.cgfs(layer, j)), inputs)) ≈ μ

    samples = RBMs.transfer_sample(layer, zeros(size(layer)..., 10^6))
    @test RBMs.transfer_mean(layer) ≈ RBMs.mean_(samples; dims=ndims(samples)) rtol=0.1
    @test RBMs.transfer_var(layer) ≈ RBMs.var_(samples;  dims=ndims(samples)) rtol=0.1
    @test RBMs.transfer_mean_abs(layer) ≈ RBMs.mean_(abs.(samples); dims=ndims(samples)) rtol=0.1
    @test all(RBMs.energy(layer, RBMs.transfer_mode(layer)) .≤ RBMs.energy(layer, samples))

    m_ex = RBMs.conjugates(layer)
    m_mc = RBMs.conjugates_empirical(layer, samples)
    @test length(m_ex) == length(m_mc)
    @test typeof(m_ex) == typeof(m_mc)
    for (∂θ_ex, ∂θ_mc) in zip(m_ex, m_mc)
        @test ∂θ_ex ≈ ∂θ_mc rtol=0.1
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
        sum(RBMs.cgfs(layer))
    end
    @test RBMs.conjugates(layer).θ ≈ gs[layer.θ] ≈ RBMs.transfer_mean(layer)
end

@testset "Binary" begin
    layer = RBMs.Binary(randn(7, 4, 5))
    @test RBMs.cgfs(layer) ≈ log.(sum(exp.(layer.θ .* h) for h in 0:1))
    @test sort(unique(RBMs.transfer_sample(layer))) == [0, 1]
end

@testset "Spin" begin
    layer = RBMs.Spin(randn(7, 4, 5))
    @test RBMs.cgfs(layer) ≈ log.(sum(exp.(layer.θ .* h) for h in (-1, 1)))
    @test sort(unique(RBMs.transfer_sample(layer))) == [-1, 1]
end

@testset "Potts" begin
    q = 3
    N = (4, 5)
    layer = RBMs.Potts(randn(q, N...))
    @test RBMs.cgfs(layer) ≈ log.(sum(exp.(layer.θ[h:h,:,:,:]) for h in 1:q))
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

    function quad_cgf(θ::Real, γ::Real)
        Z, ϵ = QuadGK.quadgk(h -> exp(-RBMs.gauss_energy(θ, γ, h)), -Inf,  Inf)
        return log(Z)
    end

    @test RBMs.cgfs(layer) ≈ quad_cgf.(layer.θ, layer.γ)

    gs = Zygote.gradient(Flux.params(layer)) do
        sum(RBMs.cgfs(layer))
    end
    μ = RBMs.transfer_mean(layer)
    ν = RBMs.transfer_var(layer)
    μ2 = @. ν + μ^2
    m = RBMs.conjugates(layer)
    @test m.θ ≈ gs[layer.θ] ≈ μ
    @test m.γ ≈ gs[layer.γ] ≈ -μ2/2
end

@testset "ReLU" begin
    N = (3, 4, 5)
    B = 7

    # bound γ away from zero to avoid numerical issues with QuadGK
    layer = RBMs.ReLU(randn(N...), rand(N...) .+ 0.5)

    x = abs.(randn(size(layer)..., B))
    @test RBMs.energies(layer, x) ≈ RBMs.energies(RBMs.Gaussian(layer.θ, layer.γ), x)

    function quad_cgf(θ::Real, γ::Real)
        Z, ϵ = QuadGK.quadgk(h -> exp(-RBMs.relu_energy(θ, γ, h)), 0,  Inf)
        return log(Z)
    end
    @test RBMs.cgfs(layer) ≈ @. quad_cgf(layer.θ, layer.γ)

    gs = Zygote.gradient(Flux.params(layer)) do
        sum(RBMs.cgfs(layer))
    end
    μ = RBMs.transfer_mean(layer)
    ν = RBMs.transfer_var(layer)
    μ2 = @. ν + μ^2
    m = RBMs.conjugates(layer)
    @test m.θ ≈ gs[layer.θ] ≈ μ
    @test m.γ ≈ gs[layer.γ] ≈ -μ2/2
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
    @test RBMs.cgfs(drelu) ≈ RBMs.cgfs(prelu) ≈ RBMs.cgfs(xrelu)
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
    @test RBMs.cgfs(drelu) ≈ RBMs.cgfs(prelu) ≈ RBMs.cgfs(xrelu)
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
    @test RBMs.cgfs(drelu) ≈ RBMs.cgfs(prelu) ≈ RBMs.cgfs(xrelu)
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
        RBMs.cgfs(gauss) ≈ RBMs.cgfs(drelu) ≈
        RBMs.cgfs(prelu) ≈ RBMs.cgfs(xrelu)
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

    relu  = RBMs.ReLU(randn(N...), rand(N...))
    drelu = @inferred RBMs.dReLU(relu)
    prelu = @inferred RBMs.pReLU(relu)
    xrelu = @inferred RBMs.xReLU(relu)
    @test (
        RBMs.energies(relu, x)  ≈ RBMs.energies(drelu, x) ≈
        RBMs.energies(prelu, x) ≈ RBMs.energies(xrelu, x)
    )
    @test (
        RBMs.cgfs(relu)  ≈ RBMs.cgfs(drelu) ≈
        RBMs.cgfs(prelu) ≈ RBMs.cgfs(xrelu)
    )
    @test (
        RBMs.transfer_mode(relu)  ≈ RBMs.transfer_mode(drelu) ≈
        RBMs.transfer_mode(prelu) ≈ RBMs.transfer_mode(xrelu)
    )
    @test (
        RBMs.transfer_mean(relu)  ≈ RBMs.transfer_mean(drelu) ≈
        RBMs.transfer_mean(prelu) ≈ RBMs.transfer_mean(xrelu)
    )
    @test (
        RBMs.transfer_mean_abs(relu)  ≈ RBMs.transfer_mean_abs(drelu) ≈
        RBMs.transfer_mean_abs(prelu) ≈ RBMs.transfer_mean_abs(xrelu)
    )
    @test (
        RBMs.transfer_var(relu)  ≈ RBMs.transfer_var(drelu) ≈
        RBMs.transfer_var(prelu) ≈ RBMs.transfer_var(xrelu)
    )
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

    function quad_cgf(θp::Real, θn::Real, γp::Real, γn::Real)
        Z, ϵ = QuadGK.quadgk(h -> exp(-RBMs.drelu_energy(θp, θn, γp, γn, h)), -Inf, Inf)
        return log(Z)
    end

    @test RBMs.cgfs(layer) ≈ quad_cgf.(layer.θp, layer.θn, abs.(layer.γp), abs.(layer.γn))

    gs = Zygote.gradient(Flux.params(layer)) do
        sum(RBMs.cgfs(layer))
    end
    m = RBMs.conjugates(layer)
    @test m.θp ≈ gs[layer.θp]
    @test m.θn ≈ gs[layer.θn]
    @test m.γp ≈ gs[layer.γp]
    @test m.γn ≈ gs[layer.γn]
end

@testset "pReLU" begin
    N = (3, 5)
    layer = RBMs.pReLU(randn(N...), rand(N...), randn(N...), 2rand(N...) .- 1)
    gs = Zygote.gradient(Flux.params(layer)) do
        sum(RBMs.cgfs(layer))
    end
    m = RBMs.conjugates(layer)
    @test m.θ ≈ gs[layer.θ]
    @test m.γ ≈ gs[layer.γ]
    @test m.Δ ≈ gs[layer.Δ]
    @test m.η ≈ gs[layer.η]
end

@testset "xReLU" begin
    N = (3, 5)
    layer = RBMs.xReLU(randn(N...), rand(N...), randn(N...), randn(N...))
    gs = Zygote.gradient(Flux.params(layer)) do
        sum(RBMs.cgfs(layer))
    end
    m = RBMs.conjugates(layer)
    @test m.θ ≈ gs[layer.θ]
    @test m.γ ≈ gs[layer.γ]
    @test m.Δ ≈ gs[layer.Δ]
    @test m.ξ ≈ gs[layer.ξ]
end
