include("tests_init.jl")

_layers = (
    RBMs.Binary,
    RBMs.Spin,
    RBMs.Potts,
    RBMs.Gaussian,
    RBMs.ReLU,
    RBMs.dReLU,
    RBMs.pReLU
)

function random_layer(
    ::Type{T}, N::Int...
) where {T <: Union{RBMs.Binary, RBMs.Spin, RBMs.Potts}}
    return T(randn(N...))
end

function random_layer(
    ::Type{T}, N::Int...
) where {T <: Union{RBMs.Gaussian, RBMs.ReLU}}
    return T(randn(N...), randn(N...))
end

function random_layer(::Type{RBMs.dReLU}, N::Int...)
    return RBMs.dReLU(randn(N...), randn(N...), randn(N...), randn(N...))
end

function random_layer(::Type{RBMs.pReLU}, N::Int...)
    return RBMs.pReLU(randn(N...), randn(N...), randn(N...), 2rand(N...) .- 1)
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
    @test size(RBMs.sample_from_inputs(layer, inputs, 1)) == size(inputs)
    @test RBMs.cgf(layer, inputs, 1) ≈ RBMs.cgf(layer, inputs)
    @inferred RBMs.energy(layer, x)
    @inferred RBMs.cgf(layer, inputs, 1)
    @inferred RBMs.sample_from_inputs(layer, inputs, 1)
end

@testset "discrete layers ($Layer)" for Layer in (RBMs.Binary, RBMs.Spin, RBMs.Potts)
    layer = random_layer(Layer, 3, 4, 5)
    x = bitrand(size(layer)..., 7)
    @test RBMs.energy(layer, x) ≈ -vec(sum(layer.θ .* x; dims=(1,2,3)))
end

@testset "Binary" begin
    N = (3, 4, 5)
    B = 7

    layer = random_layer(RBMs.Binary, N...)
    v = rand(Bool, size(layer)..., B)
    inputs = randn(size(layer)..., B)

    β = rand()

    @test RBMs.energy(layer, v) ≈ -vec(sum(layer.θ .* v; dims=(1,2,3)))
    Γ = LogExpFunctions.log1pexp.(β .* (layer.θ .+ inputs)) / β
    @test RBMs.cgf(layer, inputs, β) ≈ vec(sum(Γ; dims=(1,2,3)))

    Γ = log.(sum(exp.(β * (inputs .+ layer.θ) .* h) for h in (0, 1))) / β
    @test RBMs.cgf(layer, inputs, β) ≈ vec(sum(Γ; dims=(1,2,3)))

    @test all(RBMs.sample_from_inputs(layer, inputs) .∈ Ref(0:1))

    μ = LogExpFunctions.logistic.(β .* (layer.θ .+ inputs))
    @test only(Zygote.gradient(x -> sum(RBMs.cgf(layer, x, β)), inputs)) ≈ μ

    layer = RBMs.Binary([0.5])
    @test mean(RBMs.sample_from_inputs(layer, zeros(1, 10^6))) ≈ LogExpFunctions.logistic(0.5) rtol=0.1
end

@testset "Spin" begin
    N = (3, 4, 5)
    B = 7

    layer = random_layer(RBMs.Spin, N...)
    v = rand((-1, 1), size(layer)..., B)
    inputs = randn(size(layer)..., B)

    β = rand()

    @test RBMs.energy(layer, v) ≈ -vec(sum(layer.θ .* v; dims=(1,2,3)))
    @test RBMs.energy(layer, v) ≈ RBMs.energy(RBMs.Binary(layer.θ), v)

    Γ = LogExpFunctions.logaddexp.(β * (layer.θ .+ inputs), -β * (layer.θ .+ inputs)) / β
    @test RBMs.cgf(layer, inputs, β) ≈ vec(sum(Γ; dims=(1,2,3)))

    Γ = log.(sum(exp.(β * (inputs .* h .+ layer.θ .* h)) for h in (-1, 1))) / β
    @test RBMs.cgf(layer, inputs, β) ≈ vec(sum(Γ; dims=(1,2,3)))

    @test all(RBMs.sample_from_inputs(layer, inputs, β) .∈ Ref((-1, 1)))

    μ = tanh.(β * (layer.θ .+ inputs))
    @test only(Zygote.gradient(x -> sum(RBMs.cgf(layer, x, β)), inputs)) ≈ μ

    layer = RBMs.Spin([0.5])
    @test mean(RBMs.sample_from_inputs(layer, zeros(1, 10^6))) ≈ tanh(0.5) rtol=0.1
end

@testset "Potts" begin
    q = 3
    N = (4, 5)
    B = 7

    layer = random_layer(RBMs.Potts, q, N...)
    v = rand(Bool, q, N..., B)
    inputs = randn(q, N..., B)

    # samples are proper one-hot
    @test sort(unique(RBMs.sample_from_inputs(layer, inputs))) == [0, 1]
    @test all(sum(RBMs.sample_from_inputs(layer, inputs); dims=1) .== 1)
    @test RBMs.energy(layer, v) ≈ -vec(sum(layer.θ .* v; dims=(1,2,3)))
    @test RBMs.energy(layer, v) ≈ RBMs.energy(RBMs.Binary(layer.θ), v)

    β = rand()

    Γ = LogExpFunctions.logsumexp(β .* (layer.θ .+ inputs); dims=1) / β
    @test RBMs.cgf(layer, inputs, β) ≈ vec(sum(Γ; dims=(1,2,3)))

    Γ = log.(sum(exp.(β * (inputs .+ layer.θ)[h:h,:,:,:]) for h in 1:q)) / β
    @test RBMs.cgf(layer, inputs, β) ≈ vec(sum(Γ; dims=(1,2,3)))

    μ = LogExpFunctions.softmax(β .* (layer.θ .+ inputs); dims=1)
    @test only(Zygote.gradient(x -> sum(RBMs.cgf(layer, x, β)), inputs)) ≈ μ

    layer = RBMs.Potts([0.5; 0.2])
    μ = LogExpFunctions.softmax(layer.θ; dims=1)
    @test vec(mean(RBMs.sample_from_inputs(layer, zeros(2, 10^6)); dims=2)) ≈ μ rtol=0.1
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
    Γ = @. my_cgf(β * (inputs + layer.θ), β * abs(layer.γ)) / β
    @test RBMs.cgf(layer, inputs, β) ≈ vec(sum(Γ; dims=(1,2,3)))

    layer = RBMs.Gaussian([0.4], [0.2])
    samples = RBMs.sample_from_inputs(layer, zeros(1, 10^6))
    @test mean(samples) ≈ 2 rtol=0.1
    @test std(samples) ≈ √5 rtol=0.1
end

@testset "ReLU" begin
    N = (3, 4, 5)
    B = 7

    layer = random_layer(RBMs.ReLU, N...)
    x = randn(size(layer)..., B)
    inputs = randn(size(layer)..., B)
    xp = max.(x, 0)

    @test RBMs.energy(layer, xp) ≈ RBMs.energy(RBMs.Gaussian(layer.θ, layer.γ), xp)

    β = rand()

    function my_cgf(θ::Real, γ::Real)
        Z, ϵ = QuadGK.quadgk(h -> exp(-RBMs.relu_energy(θ, γ, h)), 0,  Inf)
        return log(Z)
    end
    Γ = @. my_cgf(β * (inputs + layer.θ), β * abs(layer.γ)) / β
    @test RBMs.cgf(layer, inputs, β) ≈ vec(sum(Γ; dims=(1,2,3)))
end

@testset "pReLU / dReLU convert" begin
    N = (10, 7)
    B = 13
    x = randn(N..., B)

    prelu = random_layer(RBMs.pReLU, N...)
    drelu = RBMs.dReLU(prelu)
    @test prelu.θ ≈ RBMs.pReLU(drelu).θ
    @test prelu.Δ ≈ RBMs.pReLU(drelu).Δ
    @test prelu.γ ≈ RBMs.pReLU(drelu).γ
    @test prelu.η ≈ RBMs.pReLU(drelu).η
    @test RBMs.energy(prelu, x) ≈ @inferred RBMs.energy(drelu, x)
    @test RBMs.cgf(prelu, x) ≈ @inferred RBMs.cgf(drelu, x)

    drelu = random_layer(RBMs.dReLU, N...)
    prelu = RBMs.pReLU(drelu)
    @test drelu.θp ≈ RBMs.dReLU(prelu).θp
    @test drelu.θn ≈ RBMs.dReLU(prelu).θn
    @test drelu.γp ≈ RBMs.dReLU(prelu).γp
    @test drelu.γn ≈ RBMs.dReLU(prelu).γn
    @test RBMs.energy(drelu, x) ≈ @inferred RBMs.energy(prelu, x)
    @test RBMs.cgf(drelu, x) ≈ @inferred RBMs.cgf(prelu, x)
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

    β = rand()
    inputs = randn(N..., B)

    function my_cgf(θp::Real, θn::Real, γp::Real, γn::Real)
        Z, ϵ = QuadGK.quadgk(h -> exp(-RBMs.drelu_energy(θp, θn, γp, γn, h)), -Inf, Inf)
        return log(Z)
    end

    Γ = @. my_cgf(β * (inputs + l.θp), β * (inputs + l.θn), β * abs(l.γp), β * abs(l.γn)) / β
    @test RBMs.cgf(l, inputs, β) ≈ vec(sum(Γ; dims=(1,2)))
end
