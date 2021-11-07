include("tests_init.jl")

@testset "testing $Layer" for Layer in (Binary, Spin, Potts, Gaussian, StdGaussian, ReLU, dReLU, pReLU)
    layer = Layer(3,4,5)

    @test (@inferred size(layer)) == (3,4,5)
    @test (@inferred length(layer)) == 3 * 4 * 5
    @test (@inferred ndims(layer)) == 3

    x = rand(Bool, size(layer)..., 7)
    @test size(@inferred energy(layer, x)) == (7,)

    inputs = randn(size(layer)..., 7)
    @test size(@inferred cgf(layer, inputs, 1.0)) == (7,)
    @test size(@inferred sample_from_inputs(layer, inputs, 1.0)) == size(inputs)

    @test cgf(layer, inputs, 1) ≈ cgf(layer, inputs)
end

@testset "fields only layers energy ($Layer)" for Layer in (Binary, Spin, Potts)
    layer = Layer(3,4,5)
    x = rand(Bool, size(layer)..., 7)
    @test energy(layer, x) ≈ -sum_(layer.θ .* x; dims=(1,2,3))
end

@testset "Binary" begin
    layer = Binary(randn(3,4,5))
    x = rand(0:1, size(layer)..., 7)
    β = rand()
    @test cgf(layer, x) ≈ sum_(log1pexp.(layer.θ .+ x); dims=(1,2,3))
    @test all(sample_from_inputs(layer, x) .∈ Ref((0, 1)))
    @test gradient(x -> sum(cgf(layer, x)), x)[1] ≈ logistic.(layer.θ .+ x)
    @test gradient(x -> sum(cgf(layer, x, β)), x)[1] ≈ logistic.(β .* (layer.θ .+ x))
    @test mean(sample_from_inputs(Binary([0.5]), zeros(1, 10^6))) ≈ logistic(0.5) rtol=0.1
end

@testset "Spin" begin
    layer = Spin(randn(3,4,5))
    x = rand((-1, 1), size(layer)..., 7)
    @test cgf(layer, x) ≈ sum_(logaddexp.(layer.θ .+ x, -layer.θ .- x); dims=(1,2,3))
    @test all(sample_from_inputs(layer, x) .∈ Ref((-1, 1)))
    @test gradient(x -> sum(cgf(layer, x)), x)[1] ≈ tanh.(layer.θ .+ x)
    β = rand()
    @test gradient(x -> sum(cgf(layer, x, β)), x)[1] ≈ tanh.(β * (layer.θ .+ x))
    @test mean(sample_from_inputs(Spin([0.5]), zeros(1, 10^6))) ≈ tanh(0.5) rtol=0.1
end

@testset "Potts" begin
    layer = Potts(randn(3,4,5))
    x = rand((-1, 1), size(layer)..., 7)
    @test cgf(layer, x) ≈ sum_(logsumexp(layer.θ .+ x; dims=1); dims=(1,2,3))
    @test all(sample_from_inputs(layer, x) .∈ Ref((0, 1)))
    @test all(sum(sample_from_inputs(layer, x); dims=1) .== 1)
    β = rand()
    @test gradient(x -> sum(cgf(layer, x, β)), x)[1] ≈ softmax(β .* (layer.θ .+ x); dims=1)
    @test mean_(sample_from_inputs(Potts([0.5; 0.2]), zeros(2, 10^6)); dims=2) ≈ softmax([0.5; 0.2]; dims=1) rtol=0.1
end

@testset "Gaussian" begin
    layer = Gaussian(randn(3,4,5), randn(3,4,5))
    x = randn(size(layer)..., 7)
    @test energy(layer, x) ≈ sum_(@. abs(layer.γ) * x^2 / 2 - layer.θ * x; dims=(1,2,3))
    @test cgf(layer, x) ≈ sum_(@. (layer.θ + x)^2 / abs(2layer.γ) + log(2π/abs(layer.γ)) / 2; dims=(1,2,3))
    @test gradient(x -> sum(cgf(layer, x)), x)[1] ≈ @. (layer.θ + x)/abs(layer.γ)
    β = rand()
    @test gradient(x -> sum(cgf(layer, x, β)), x)[1] ≈ @. (layer.θ + x)/abs(layer.γ)

    layer = Gaussian([0.4], [0.2])
    samples = sample_from_inputs(layer, zeros(1, 10^6))
    @test mean(samples) ≈ 0.4 / 0.2 rtol=0.1
    @test std(samples) ≈ 1 / √0.2 rtol=0.1
end

@testset "pReLU / dReLU convert" begin
    layer = pReLU(randn(10), randn(10), rand(10), rand(10) .- 0.5)
    layer_ = pReLU(dReLU(layer))
    @test layer.θ ≈ layer_.θ
    @test layer.Δ ≈ layer_.Δ
    @test layer.γ ≈ layer_.γ
    @test layer.η ≈ layer_.η

    layer = dReLU(randn(10), randn(10), rand(10), rand(10))
    layer_ = dReLU(pReLU(layer))
    @test layer.θp ≈ layer_.θp
    @test layer.θn ≈ layer_.θn
    @test layer.γp ≈ layer_.γp
    @test layer.γn ≈ layer_.γn
end
