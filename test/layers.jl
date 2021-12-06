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

@testset "fields only layers ($Layer)" for Layer in (RBMs.Binary, RBMs.Spin, RBMs.Potts)
    layer = Layer(3,4,5)
    x = bitrand(size(layer)..., 7)
    @test RBMs.energy(layer, x) ≈ -vec(sum(layer.θ .* x; dims=(1,2,3)))
end

@testset "Binary" begin
    layer = RBMs.Binary(randn(3,4,5))
    v = bitrand(size(layer)..., 7)
    @test RBMs.energy(layer, v) ≈ -vec(sum(layer.θ .* v; dims=(1,2,3)))
    @test RBMs.cgf(layer, v) ≈ vec(sum(log1pexp.(layer.θ .+ v); dims=(1,2,3)))
    @test all(RBMs.sample_from_inputs(layer, v) .∈ Ref(0:1))
    @test gradient(x -> sum(RBMs.cgf(layer, x)), float(v))[1] ≈ logistic.(layer.θ .+ v)
    β = rand()
    @test gradient(x -> sum(RBMs.cgf(layer, x, β)), float(v))[1] ≈ logistic.(β .* (layer.θ .+ v))
    @test mean(RBMs.sample_from_inputs(RBMs.Binary([0.5]), zeros(1, 10^6))) ≈ logistic(0.5) rtol=0.1
end

@testset "Spin" begin
    layer = RBMs.Spin(randn(3,4,5))
    v = rand((-1, +1), size(layer)..., 7)
    @test RBMs.energy(layer, v) ≈ -vec(sum(layer.θ .* v; dims=(1,2,3)))

    Γ = vec(sum(LogExpFunctions.logaddexp.(layer.θ .+ v, -layer.θ .- v); dims=(1,2,3)))
    @test RBMs.cgf(layer, v) ≈ Γ

    @test all(RBMs.sample_from_inputs(layer, v) .∈ Ref((-1, 1)))
    @test gradient(x -> sum(RBMs.cgf(layer, x)), v)[1] ≈ tanh.(layer.θ .+ v)

    β = rand()
    μ = tanh.(β * (layer.θ .+ v))
    @test gradient(x -> sum(RBMs.cgf(layer, x, β)), v)[1] ≈ μ

    layer = RBMs.Spin([0.5])
    @test mean(RBMs.sample_from_inputs(layer, zeros(1, 10^6))) ≈ tanh(0.5) rtol=0.1
end

@testset "Potts" begin
    layer = RBMs.Potts(randn(3,4,5))
    v = bitrand(size(layer)..., 7)
    @test sort(unique(RBMs.sample_from_inputs(layer, v))) == [0, 1]
    @test all(sum(RBMs.sample_from_inputs(layer, v); dims=1) .== 1)

    Γ = vec(sum(LogExpFunctions.logsumexp(layer.θ .+ v; dims=1); dims=(1,2,3)))
    @test RBMs.cgf(layer, v) ≈ Γ

    β = rand()

    μ = LogExpFunctions.softmax(β .* (layer.θ .+ v); dims=1)
    @test gradient(x -> sum(RBMs.cgf(layer, x, β)), v)[1] ≈ μ

    layer = RBMs.Potts([0.5; 0.2])
    μ = LogExpFunctions.softmax(layer.θ; dims=1)
    @test vec(mean(RBMs.sample_from_inputs(layer, zeros(2, 10^6)); dims=2)) ≈ μ rtol=0.1
end

@testset "Gaussian" begin
    layer = RBMs.Gaussian(randn(3,4,5), randn(3,4,5))
    x = randn(size(layer)..., 7)
    E = @. abs(layer.γ) * x^2 / 2 - layer.θ * x
    Γ = @. (layer.θ + x)^2 / abs(2layer.γ) + log(2π / abs(layer.γ)) / 2

    @test RBMs.energy(layer, x) ≈ vec(sum(E; dims=(1,2,3)))
    @test RBMs.cgf(layer, x) ≈ vec(sum(Γ; dims=(1,2,3)))

    μ = @. (layer.θ + x) / abs(layer.γ)
    @test gradient(x -> sum(RBMs.cgf(layer, x)), x)[1] ≈ μ

    β = rand()
    @test gradient(x -> sum(RBMs.cgf(layer, x, β)), x)[1] ≈ μ

    layer = RBMs.Gaussian([0.4], [0.2])
    samples = RBMs.sample_from_inputs(layer, zeros(1, 10^6))
    @test mean(samples) ≈ 0.4 / 0.2 rtol=0.1
    @test std(samples) ≈ 1 / √0.2 rtol=0.1
end

@testset "pReLU / dReLU convert" begin
    layer = RBMs.pReLU(randn(10), randn(10), rand(10), rand(10) .- 0.5)
    layer_ = RBMs.pReLU(RBMs.dReLU(layer))
    @test layer.θ ≈ layer_.θ
    @test layer.Δ ≈ layer_.Δ
    @test layer.γ ≈ layer_.γ
    @test layer.η ≈ layer_.η

    layer = RBMs.dReLU(randn(10), randn(10), rand(10), rand(10))
    layer_ = RBMs.dReLU(RBMs.pReLU(layer))
    @test layer.θp ≈ layer_.θp
    @test layer.θn ≈ layer_.θn
    @test layer.γp ≈ layer_.γp
    @test layer.γn ≈ layer_.γn
end
