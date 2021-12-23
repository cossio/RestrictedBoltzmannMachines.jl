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

function random_layer(
    ::Type{T}, N::Int...
) where {T <: Union{RBMs.Gaussian, RBMs.ReLU}}
    return T(randn(N...), rand(N...))
end

function random_layer(::Type{RBMs.dReLU}, N::Int...)
    return RBMs.dReLU(randn(N...), randn(N...), rand(N...), rand(N...))
end

function random_layer(::Type{RBMs.xReLU}, N::Int...)
    return RBMs.xReLU(randn(N...), randn(N...), rand(N...), randn(N...))
end

function random_layer(::Type{RBMs.pReLU}, N::Int...)
    return RBMs.pReLU(randn(N...), randn(N...), rand(N...), 2rand(N...) .- 1)
end

@testset "testing $Layer" for Layer in _layers
    layer = random_layer(Layer, 3, 4, 5)

    @test (@inferred size(layer)) == (3,4,5)
    @test (@inferred length(layer)) == 3 * 4 * 5
    @test (@inferred ndims(layer)) == 3

    x = rand(Bool, size(layer)..., 7)
    @test size(RBMs.energy(layer, x)) == (7,)

    inputs = randn(size(layer)..., 7)
    @test size(RBMs.cgf(layer, inputs, 1)) == (7,)
    @test size(RBMs.transfer_sample(layer, inputs, 1)) == size(inputs)
    @test size(RBMs.transfer_sample(layer, 0, 1)) == size(RBMs.transfer_sample(layer)) == size(layer)
    @test RBMs.cgf(layer, inputs, 1) ≈ RBMs.cgf(layer, inputs)
    @test RBMs.cgfs(layer, 0, 1) ≈ RBMs.cgfs(layer)
    @test RBMs.cgf(layer) isa Real
    @inferred RBMs.energy(layer, x)
    @inferred RBMs.cgfs(layer, inputs, 1)
    @inferred RBMs.cgf(layer, inputs, 1)
    @inferred RBMs.transfer_sample(layer, inputs, 1)
end

@testset "discrete layers ($Layer)" for Layer in (RBMs.Binary, RBMs.Spin, RBMs.Potts)
    layer = random_layer(Layer, 3, 4, 5)
    x = bitrand(size(layer)..., 7)
    @test RBMs.energy(layer, x) ≈ -vec(sum(layer.θ .* x; dims=(1,2,3)))
end

@testset "Binary" begin
    N = (3, 4, 5)
    B = 7

    layer = RBMs.Binary(randn(N...))
    x = rand(Bool, size(layer)..., B)
    inputs = randn(size(layer)..., B)

    @test RBMs.energies(layer, x) ≈ -layer.θ .* x
    @test RBMs.energy(layer, x) ≈ -vec(sum(layer.θ .* x; dims=(1,2,3)))

    β = rand()

    Γ = log.(sum(exp.(β * (inputs .+ layer.θ) .* h) for h in (0, 1))) / β
    @test RBMs.cgfs(layer, inputs, β) ≈ Γ
    @test RBMs.cgf(layer, inputs, β) ≈ vec(sum(Γ; dims=(1,2,3)))

    @test all(RBMs.transfer_sample(layer, inputs, β) .∈ Ref(0:1))

    μ = LogExpFunctions.logistic.(β .* (layer.θ .+ inputs))
    @test only(Zygote.gradient(x -> sum(RBMs.cgfs(layer, x, β)), inputs)) ≈ μ

    layer = RBMs.Binary([0.5])
    @test mean(RBMs.transfer_sample(layer, zeros(1, 10^6))) ≈ LogExpFunctions.logistic(0.5) rtol=0.1
    @test mean(RBMs.transfer_sample(layer, zeros(1, 10^6))) ≈ only(RBMs.transfer_mean(layer)) rtol=0.1
end

@testset "Spin" begin
    N = (3, 4, 5)
    B = 7

    layer = RBMs.Spin(randn(N...))
    v = rand((-1, 1), size(layer)..., B)
    inputs = randn(size(layer)..., B)

    β = rand()

    @test RBMs.energy(layer, v) ≈ -vec(sum(layer.θ .* v; dims=(1,2,3)))
    @test RBMs.energy(layer, v) ≈ RBMs.energy(RBMs.Binary(layer.θ), v)

    Γ = log.(sum(exp.(β * (inputs .+ layer.θ) .* h) for h in (-1, 1))) / β
    @test RBMs.cgfs(layer, inputs, β) ≈ Γ
    @test RBMs.cgf(layer, inputs, β) ≈ vec(sum(Γ; dims=(1,2,3)))

    @test all(RBMs.transfer_sample(layer, inputs, β) .∈ Ref((-1, 1)))

    μ = tanh.(β * (layer.θ .+ inputs))
    @test only(Zygote.gradient(x -> sum(RBMs.cgf(layer, x, β)), inputs)) ≈ μ

    layer = RBMs.Spin([0.5])
    @test mean(RBMs.transfer_sample(layer, zeros(1, 10^6))) ≈ tanh(0.5) rtol=0.1
    @test mean(RBMs.transfer_sample(layer, zeros(1, 10^6))) ≈ only(RBMs.transfer_mean(layer)) rtol=0.1
end

@testset "Potts" begin
    q = 3
    N = (4, 5)
    B = 7

    layer = random_layer(RBMs.Potts, q, N...)
    v = rand(Bool, q, N..., B)
    inputs = randn(q, N..., B)

    # samples are proper one-hot
    @test sort(unique(RBMs.transfer_sample(layer, inputs))) == [0, 1]
    @test all(sum(RBMs.transfer_sample(layer, inputs); dims=1) .== 1)
    @test RBMs.energy(layer, v) ≈ -vec(sum(layer.θ .* v; dims=(1,2,3)))
    @test RBMs.energy(layer, v) ≈ RBMs.energy(RBMs.Binary(layer.θ), v)

    β = rand()

    Γ = LogExpFunctions.logsumexp(β .* (layer.θ .+ inputs); dims=1) / β
    @test RBMs.cgf(layer, inputs, β) ≈ vec(sum(Γ; dims=(1,2,3)))

    Γ = log.(sum(exp.(β * (inputs .+ layer.θ)[h:h,:,:,:]) for h in 1:q)) / β
    @test RBMs.cgfs(layer, inputs, β) ≈ Γ
    @test RBMs.cgf(layer, inputs, β) ≈ vec(sum(Γ; dims=(1,2,3)))

    μ = LogExpFunctions.softmax(β .* (layer.θ .+ inputs); dims=1)
    @test only(Zygote.gradient(x -> sum(RBMs.cgf(layer, x, β)), inputs)) ≈ μ

    layer = RBMs.Potts([0.5; 0.2])
    μ = LogExpFunctions.softmax(layer.θ; dims=1)
    @test vec(mean(RBMs.transfer_sample(layer, zeros(2, 10^6)); dims=2)) ≈ μ rtol=0.1
end

@testset "Gaussian" begin
    N = (3, 4, 5)
    B = 7

    layer = random_layer(RBMs.Gaussian, N...)
    v = randn(size(layer)..., B)
    inputs = randn(size(layer)..., B)

    β = rand()

    E = @. abs(layer.γ) * v^2 / 2 - layer.θ * v
    Γ = @. (layer.θ + inputs)^2 / abs(2layer.γ) + log(2π / abs(β * layer.γ)) / 2β

    @test RBMs.energy(layer, v) ≈ vec(sum(E; dims=(1,2,3)))
    @test RBMs.cgf(layer, inputs, β) ≈ vec(sum(Γ; dims=(1,2,3)))

    μ = @. (layer.θ + inputs) / abs(layer.γ)
    @test only(Zygote.gradient(x -> sum(RBMs.cgf(layer, x, β)), inputs)) ≈ μ

    function my_cgf(θ::Real, γ::Real)
        Z, ϵ = QuadGK.quadgk(h -> exp(θ * h - γ * h^2 / 2), -Inf, Inf)
        return log(Z)
    end
    # bound γ away from zero to avoid issues with QuadGK
    layer = RBMs.Gaussian(randn(N...), 0.5 .+ rand(N...))
    β = 1.5
    Γ = @. my_cgf(β * (inputs + layer.θ), β * abs(layer.γ)) / β
    @test RBMs.cgf(layer, inputs, β) ≈ vec(sum(Γ; dims=(1,2,3)))

    layer = RBMs.Gaussian([0.4], [0.2])
    samples = RBMs.transfer_sample(layer, zeros(1, 10^6))
    @test mean(samples) ≈ 2 rtol=0.1
    @test std(samples) ≈ √5 rtol=0.1

    N = (16, 8)
    B = 64
    l = random_layer(RBMs.Gaussian, N...)
    inputs = randn(N..., B)
    x = randn(N..., B)
    m = RBMs.transfer_mode(l, inputs)
    @test all(inputs .* m .- RBMs.energies(l, m) .≥ inputs .* x .- RBMs.energies(l, x))
end

@testset "ReLU" begin
    N = (3, 4, 5)
    B = 7

    layer = random_layer(RBMs.ReLU, N...)
    x = randn(size(layer)..., B)
    inputs = randn(size(layer)..., B)
    xp = max.(x, 0)

    @test RBMs.energy(layer, xp) ≈ RBMs.energy(RBMs.Gaussian(layer.θ, layer.γ), xp)

    function my_cgf(θ::Real, γ::Real)
        Z, ϵ = QuadGK.quadgk(h -> exp(-RBMs.relu_energy(θ, γ, h)), 0,  Inf)
        return log(Z)
    end
    # bound γ away from zero to avoid issues with QuadGK
    layer = RBMs.ReLU(randn(N...), rand(N...) .+ 1)
    β = 1.5
    Γ = @. my_cgf(β * (inputs + layer.θ), β * abs(layer.γ)) / β
    @test RBMs.cgf(layer, inputs, β) ≈ vec(sum(Γ; dims=(1,2,3)))
end

@testset "pReLU / xReLU / dReLU convert" begin
    N = (10, 7)
    B = 13
    x = randn(N..., B)

    drelu = random_layer(RBMs.dReLU, N...)
    prelu = @inferred RBMs.pReLU(drelu)
    xrelu = @inferred RBMs.xReLU(drelu)
    @test drelu.θp ≈ RBMs.dReLU(prelu).θp ≈ RBMs.dReLU(xrelu).θp
    @test drelu.θn ≈ RBMs.dReLU(prelu).θn ≈ RBMs.dReLU(xrelu).θn
    @test drelu.γp ≈ RBMs.dReLU(prelu).γp
    @test drelu.γn ≈ RBMs.dReLU(prelu).γn
    @test abs.(drelu.γp) ≈ RBMs.dReLU(xrelu).γp
    @test abs.(drelu.γn) ≈ RBMs.dReLU(xrelu).γn
    @test RBMs.energy(drelu, x) ≈ RBMs.energy(prelu, x) ≈ RBMs.energy(xrelu, x)
    @test RBMs.cgf(drelu, x) ≈ RBMs.cgf(prelu, x) ≈ RBMs.cgf(xrelu, x)

    prelu = random_layer(RBMs.pReLU, N...)
    drelu = @inferred RBMs.dReLU(prelu)
    xrelu = @inferred RBMs.xReLU(prelu)
    @test prelu.θ ≈ RBMs.pReLU(drelu).θ ≈ RBMs.pReLU(xrelu).θ
    @test prelu.Δ ≈ RBMs.pReLU(drelu).Δ ≈ RBMs.pReLU(xrelu).Δ
    @test prelu.γ ≈ RBMs.pReLU(drelu).γ ≈ RBMs.pReLU(xrelu).γ
    @test prelu.η ≈ RBMs.pReLU(drelu).η ≈ RBMs.pReLU(xrelu).η
    @test RBMs.energy(drelu, x) ≈ RBMs.energy(prelu, x) ≈ RBMs.energy(xrelu, x)
    @test RBMs.cgf(drelu, x) ≈ RBMs.cgf(prelu, x) ≈ RBMs.cgf(xrelu, x)

    xrelu = random_layer(RBMs.xReLU, N...)
    drelu = @inferred RBMs.dReLU(xrelu)
    prelu = @inferred RBMs.pReLU(xrelu)
    @test xrelu.θ ≈ RBMs.xReLU(drelu).θ ≈ RBMs.xReLU(prelu).θ
    @test xrelu.Δ ≈ RBMs.xReLU(drelu).Δ ≈ RBMs.xReLU(prelu).Δ
    @test xrelu.ξ ≈ RBMs.xReLU(drelu).ξ ≈ RBMs.xReLU(prelu).ξ
    @test xrelu.γ ≈ RBMs.xReLU(prelu).γ
    @test abs.(xrelu.γ) ≈ RBMs.xReLU(drelu).γ
    @test RBMs.energy(drelu, x) ≈ RBMs.energy(prelu, x) ≈ RBMs.energy(xrelu, x)
    @test RBMs.cgf(drelu, x) ≈ RBMs.cgf(prelu, x) ≈ RBMs.cgf(xrelu, x)

    for layer in (drelu, prelu, xrelu)
        @inferred RBMs.energy(layer, x)
        @inferred RBMs.cgf(layer, x)
    end
end

@testset "dReLU" begin
    N = (10, 2)
    B = 13
    x = randn(N..., B)
    xp = max.(x, 0)
    xn = min.(x, 0)

    l = random_layer(RBMs.dReLU, N...)

    E = RBMs.energy(RBMs.ReLU(l.θp, l.γp), xp) + RBMs.energy(RBMs.ReLU(-l.θn, l.γn), -xn)
    @test RBMs.energy(l, x) ≈ E
    @test iszero(RBMs.energy(l, zero(x)))

    function my_cgf(θp::Real, θn::Real, γp::Real, γn::Real)
        Z, ϵ = QuadGK.quadgk(h -> exp(-RBMs.drelu_energy(θp, θn, γp, γn, h)), -Inf, Inf)
        return log(Z)
    end

    # bound γ away from zero to avoid issues with QuadGK
    β = rand() + 0.5
    l = RBMs.dReLU(randn(N...), randn(N...), rand(N...) .+ 0.5, rand(N...) .+ 0.5)
    inputs = randn(N..., B)

    Γ = @. my_cgf(β * (inputs + l.θp), β * (inputs + l.θn), β * abs(l.γp), β * abs(l.γn)) / β
    @test RBMs.cgf(l, inputs, β) ≈ vec(sum(Γ; dims=(1,2)))

    N = (16, 8)
    B = 64
    l = RBMs.dReLU(randn(N...), randn(N...), rand(N...) .+ 0.5, rand(N...) .+ 0.5)
    inputs = randn(N..., B)
    x = randn(N..., B)
    m = RBMs.transfer_mode(l, inputs)
    @test all(inputs .* m .- RBMs.energies(l, m) .≥ inputs .* x .- RBMs.energies(l, x))
end
