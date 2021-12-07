include("tests_init.jl")

_layers = (
    RBMs.Binary,
    RBMs.Spin,
    RBMs.Potts,
    RBMs.Gaussian,
    RBMs.StdGaussian,
    RBMs.ReLU,
    RBMs.dReLU,
    RBMs.pReLU
)

@testset "testing $Layer" for Layer in _layers
    layer = Layer(3,4,5)

    @test (@inferred size(layer)) == (3,4,5)
    @test (@inferred length(layer)) == 3 * 4 * 5
    @test (@inferred ndims(layer)) == 3

    x = rand(Bool, size(layer)..., 7)
    @test size(@inferred RBMs.energy(layer, x)) == (7,)

    inputs = randn(size(layer)..., 7)
    @test size(@inferred RBMs.cgf(layer, inputs, 1.0)) == (7,)
    @test size(@inferred RBMs.sample_from_inputs(layer, inputs, 1.0)) == size(inputs)

    @test RBMs.cgf(layer, inputs, 1) ≈ RBMs.cgf(layer, inputs)
end

@testset "discrete layers ($Layer)" for Layer in (RBMs.Binary, RBMs.Spin, RBMs.Potts)
    layer = Layer(3,4,5)
    x = bitrand(size(layer)..., 7)
    @test RBMs.energy(layer, x) ≈ -vec(sum(layer.θ .* x; dims=(1,2,3)))
end

@testset "Binary" begin
    layer = RBMs.Binary(randn(3,4,5))
    v = rand(0.0:1.0, size(layer)..., 7)
    @test RBMs.energy(layer, v) ≈ -vec(sum(layer.θ .* v; dims=(1,2,3)))

    Γ = LogExpFunctions.log1pexp.(layer.θ .+ v)
    @test RBMs.cgf(layer, v) ≈ vec(sum(Γ; dims=(1,2,3)))

    @test all(RBMs.sample_from_inputs(layer, v) .∈ Ref(0:1))

    μ = LogExpFunctions.logistic.(layer.θ .+ v)
    @test only(Zygote.gradient(x -> sum(RBMs.cgf(layer, x)), v)) ≈ μ

    β = rand()
    μ = LogExpFunctions.logistic.(β .* (layer.θ .+ v))
    @test only(Zygote.gradient(x -> sum(RBMs.cgf(layer, x, β)), v)) ≈ μ

    layer = RBMs.Binary([0.5])
    @test mean(RBMs.sample_from_inputs(layer, zeros(1, 10^6))) ≈ LogExpFunctions.logistic(0.5) rtol=0.1
end

@testset "Spin" begin
    layer = RBMs.Spin(randn(3,4,5))
    v = rand((-1.0, +1.0), size(layer)..., 7)
    @test RBMs.energy(layer, v) ≈ -vec(sum(layer.θ .* v; dims=(1,2,3)))

    Γ = vec(sum(LogExpFunctions.logaddexp.(layer.θ .+ v, -layer.θ .- v); dims=(1,2,3)))
    @test RBMs.cgf(layer, v) ≈ Γ

    @test all(RBMs.sample_from_inputs(layer, v) .∈ Ref((-1, 1)))
    @test only(Zygote.gradient(x -> sum(RBMs.cgf(layer, x)), v)) ≈ tanh.(layer.θ .+ v)

    β = rand()
    μ = tanh.(β * (layer.θ .+ v))
    @test only(Zygote.gradient(x -> sum(RBMs.cgf(layer, x, β)), v)) ≈ μ

    layer = RBMs.Spin([0.5])
    @test mean(RBMs.sample_from_inputs(layer, zeros(1, 10^6))) ≈ tanh(0.5) rtol=0.1
end

@testset "Potts" begin
    layer = RBMs.Potts(randn(3,4,5))
    v = rand(0.0:1.0, size(layer)..., 7)
    # samples are proper one-hot
    @test sort(unique(RBMs.sample_from_inputs(layer, v))) == [0, 1]
    @test all(sum(RBMs.sample_from_inputs(layer, v); dims=1) .== 1)
    @test RBMs.energy(layer, v) ≈ -vec(sum(layer.θ .* v; dims=(1,2,3)))
    @test RBMs.energy(layer, v) ≈ RBMs.energy(RBMs.Binary(layer.θ), v)

    Γ = vec(sum(LogExpFunctions.logsumexp(layer.θ .+ v; dims=1); dims=(1,2,3)))
    @test RBMs.cgf(layer, v) ≈ Γ

    β = rand()

    μ = LogExpFunctions.softmax(β .* (layer.θ .+ v); dims=1)
    @test only(Zygote.gradient(x -> sum(RBMs.cgf(layer, x, β)), v)) ≈ μ

    layer = RBMs.Potts([0.5; 0.2])
    μ = LogExpFunctions.softmax(layer.θ; dims=1)
    @test vec(mean(RBMs.sample_from_inputs(layer, zeros(2, 10^6)); dims=2)) ≈ μ rtol=0.1
end

@testset "Gaussian" begin
    layer = RBMs.Gaussian(randn(3,4,5), randn(3,4,5))
    v = randn(size(layer)..., 7)
    E = @. abs(layer.γ) * v^2 / 2 - layer.θ * v
    Γ = @. (layer.θ + v)^2 / abs(2layer.γ) + log(2π / abs(layer.γ)) / 2

    @test RBMs.energy(layer, v) ≈ vec(sum(E; dims=(1,2,3)))
    @test RBMs.cgf(layer, v) ≈ vec(sum(Γ; dims=(1,2,3)))

    μ = @. (layer.θ + v) / abs(layer.γ)
    @test only(Zygote.gradient(x -> sum(RBMs.cgf(layer, x)), v)) ≈ μ

    β = rand()
    @test only(Zygote.gradient(x -> sum(RBMs.cgf(layer, x, β)), v)) ≈ μ

    layer = RBMs.Gaussian([0.4], [0.2])
    samples = RBMs.sample_from_inputs(layer, zeros(1, 10^6))
    @test mean(samples) ≈ 0.4 / 0.2 rtol=0.1
    @test std(samples) ≈ 1 / √0.2 rtol=0.1
end

@testset "pReLU / dReLU convert" begin
    N = 10
    B = 13
    x = randn(N, B)

    layer = RBMs.pReLU(randn(N), randn(N), randn(N), randn(N))
    @test layer.θ ≈ RBMs.pReLU(RBMs.dReLU(layer)).θ
    @test layer.Δ ≈ RBMs.pReLU(RBMs.dReLU(layer)).Δ
    @test layer.γ ≈ RBMs.pReLU(RBMs.dReLU(layer)).γ
    @test layer.η ≈ RBMs.pReLU(RBMs.dReLU(layer)).η
    @test RBMs.energy(layer, x) ≈ RBMs.energy(RBMs.dReLU(layer), x)

    layer = RBMs.dReLU(randn(N), randn(N), randn(N), randn(N))
    @test layer.θp ≈ RBMs.dReLU(RBMs.pReLU(layer)).θp
    @test layer.θn ≈ RBMs.dReLU(RBMs.pReLU(layer)).θn
    @test layer.γp ≈ RBMs.dReLU(RBMs.pReLU(layer)).γp
    @test layer.γn ≈ RBMs.dReLU(RBMs.pReLU(layer)).γn
    @test RBMs.energy(layer, x) ≈ RBMs.energy(RBMs.pReLU(layer), x)
end

@testset "dReLU" begin
    N = 10
    B = 13
    x = randn(N, B)
    xp = max.(x, 0)
    xn = min.(x, 0)

    layer = RBMs.dReLU(randn(N), randn(N), rand(N), rand(N))
    E = @. abs(layer.γp) * xp^2 / 2 + abs(layer.γn) * xn^2 / 2 - layer.θp * xp - layer.θn * xn
    @test RBMs.energy(layer, x) ≈ vec(sum(E; dims=1))
end
