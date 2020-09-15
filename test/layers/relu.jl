using Test, Statistics, Random, LinearAlgebra
using Zygote, SpecialFunctions, Flux, Distributions, FiniteDifferences, ProgressMeter
using Base: front, tail
using Flux: params
using RestrictedBoltzmannMachines
using RestrictedBoltzmannMachines: ∇relu_rand, ∇relu_cgf, ∇drelu_rand,
    relu_pdf, relu_cgf, exp_relu_cgf, relu_survival

include("../test_utils.jl")

Random.seed!(569)

#= ReLU tests =#

(θ, γ) = (randn(), rand())
@test exp_relu_cgf(θ, γ) ≈ exp(relu_cgf(θ, γ))
dθ, dγ = gradient(relu_cgf, θ, γ)
Γ_, dθ_, dγ_ = ∇relu_cgf(θ, γ)
@test Γ_ ≈ relu_cgf(θ, γ)
@test dθ_ ≈ dθ
@test dγ_ ≈ dγ
Δ = central_fdm(5,1)(0) do ϵ
    relu_cgf(θ + π * ϵ, γ + ϵ)
end
@test Δ ≈ π * dθ + dγ

dθ, dγ = gradient(θ, γ) do θ, γ
    relu_cgf(θ, γ)
end
@test   dθ ≈ first(transfer_mean(ReLU([θ], [γ])))
@test -2dγ ≈ first(transfer_var(ReLU([θ], [γ]))) + dθ^2

x, dθ_, dγ_ = ∇relu_rand(θ, γ)
dθ, dγ, dx = gradient(relu_survival, θ, γ, x)
@test relu_pdf(θ, γ, x) ≈ -dx
@test dθ_ ≈ dθ / relu_pdf(θ, γ, x)
@test dγ_ ≈ dγ / relu_pdf(θ, γ, x)

layer = ReLU(fill(θ, 10^6), fill(γ, 10^6))
x, dθ_, dγ_ = ∇relu_rand(layer.θ, layer.γ)
@test mean(x) ≈ mean(transfer_mean(layer)) atol=0.01
dθ, dγ = gradient((θ,γ) -> mean(transfer_mean(ReLU([θ],[γ]))), θ, γ)
@test mean(dθ_) ≈ dθ atol=0.01
@test mean(dγ_) ≈ dγ atol=0.01

layer = ReLU(randn(2), rand(2))
ps = Flux.params(layer)
gs = gradient(ps) do
    cgf(layer, 0)
end
@test   gs[layer.θ] ≈ transfer_mean(layer)
@test -2gs[layer.γ] ≈ transfer_var(layer) .+ transfer_mean(layer).^2

dists = truncated.(Normal.(transfer_mean(Gaussian(layer)), transfer_std(Gaussian(layer))), 0, Inf)
@test mean.(dists) ≈ transfer_mean(layer)
@test std.(dists) ≈ transfer_std(layer)
@test var.(dists) ≈ transfer_var(layer)

layer = ReLU(randn(5,5), rand(5,5))
sample = random(layer, zeros(size(layer)..., 10^6))
@test transfer_mean(layer) ≈ mean(sample; dims=3) rtol=0.05
@test transfer_std(layer) ≈ std(sample; dims=3) rtol=0.05
@test transfer_var(layer) ≈ var(sample; dims=3) rtol=0.05
@test transfer_mean_abs(layer) ≈ mean(abs, sample; dims=3) rtol=0.05
sample = nothing

@testset "ReLU pdf, cdf" begin
    layer = ReLU(randn(5,5), randn(5,5))
    h = random(layer)
    gs = gradient(params(h)) do
        sum(transfer_cdf(layer, h))
    end
    @test gs[h] ≈ transfer_pdf(layer, h)
end

@testset "relu_cgf" begin
    for _ = 1:10
        # scalar version
        gradtest(relu_cgf, randn(), rand()) # scalar version
        # broadcasted version
        gradtest((θ, γ) -> relu_cgf.(θ, γ), randn(3,4), rand(3,4)) # broadcasted
    end
end

@testset "ReLU energy & cgf gradients" begin
    for _ = 1:10
        θ, γ = randn(2,3), randn(2,3)
        # with batch dimensions
        x = rand(2,3, 1,2)
        I = randn(2,3, 1,2)
        gradtest((θ, γ) -> energy(ReLU(θ, γ), x), θ, γ)
        gradtest((θ, γ) -> cgf(ReLU(θ, γ), I), θ, γ)
        (dI,) = gradient(I -> sum(cgf(ReLU(θ, γ), I)), I)
        @test dI ≈ transfer_mean(ReLU(θ, γ), I)
        # without batch dimensions
        x = rand(2,3)
        I = randn(2,3)
        gradtest((θ, γ) -> energy(ReLU(θ, γ), x), θ, γ)
        gradtest((θ, γ) -> cgf(ReLU(θ, γ), x), θ, γ)
        (dI,) = gradient(I -> cgf(ReLU(θ, γ), I), I)
        @test dI ≈ transfer_mean(ReLU(θ, γ), I)
    end
end

@testset "ReLU random gradient" begin
    layer = ReLU(randn(2,2), randn(2,2))
    ps = Flux.params(layer)
    h = random(layer, zeros(size(layer)...))
    gs = gradient(ps) do
        Zygote.@ignore Random.seed!(1)
        h_ = random(layer, zeros(size(layer)...))
        Zygote.@ignore h .= h_
        sum(h_)
    end
    ∇cdf = gradient(ps) do
        sum(transfer_cdf(layer, h))
    end
    @test gs[layer.θ] ≈ -∇cdf[layer.θ] ./ transfer_pdf(layer, h)
    @test gs[layer.γ] ≈ -∇cdf[layer.γ] ./ transfer_pdf(layer, h)
end

@testset "ReLU, contrastive divergence gradient" begin
    rbm = RBM(Potts(5,10), ReLU(randn(10,5), randn(10,5)))
    randn!(rbm.weights); randn!(rbm.vis.θ);
    randn!(rbm.hid.θ); rand!(rbm.hid.γ);
    ps = Flux.params(rbm)
    @test rbm.weights ∈ ps
    @test rbm.vis.θ ∈ ps
    @test rbm.hid.θ ∈ ps
    @test rbm.hid.γ ∈ ps
    vd = random(rbm.vis, zeros(size(rbm.vis, 10)))
    vm = random(rbm.vis, zeros(size(rbm.vis, 10)))
    wd = rand(10)
    gs = gradient(ps) do
        contrastive_divergence(rbm, vd, vm, wd)
    end
    pw = randn(size(rbm.weights)); pg = randn(size(rbm.vis));
    pθ = randn(size(rbm.hid)); pγ = randn(size(rbm.hid));
    Δ = central_fdm(5,1)(0) do ϵ
        rbm_ = RBM(Potts(rbm.vis.θ .+ ϵ .* pg),
                   ReLU(rbm.hid.θ .+ ϵ .* pθ, rbm.hid.γ .+ ϵ .* pγ),
                   rbm.weights .+ ϵ .* pw)
        contrastive_divergence(rbm_, vd, vm, wd)
    end
    @test Δ ≈ dot(gs[rbm.weights], pw) + dot(gs[rbm.vis.θ], pg) +
              dot(gs[rbm.hid.θ], pθ) + dot(gs[rbm.hid.γ], pγ)
end

@testset "ReLU cd training" begin
    V = Binary; H = ReLU;

    teacher = RBM(V(7), H(2));
    randn!(teacher.weights)
    teacher.weights .*= 5/sqrt(length(teacher.vis))
    v = zeros(size(teacher.vis)..., 1000)
    @showprogress for t = 1:1000
        h = sample_h_from_v(teacher, v)
        v .= sample_v_from_h(teacher, h)
    end
    data = Data((v = v, w = ones(size(v)[end])), batchsize=64)
    log_pseudolikelihood_rand(teacher, data.tensors.v)
    log_likelihood(teacher, data.tensors.v) |> mean

    student = RBM(V(size(teacher.vis)...), H(size(teacher.hid)...))
    log_likelihood(student, data.tensors.v) |> mean

    cor(free_energy(teacher, data.tensors.v),
        free_energy(student, data.tensors.v))

    #init!(student, data.tensors.v; eps=1e-10)
    student.weights .= 0
    log_pseudolikelihood_rand(student, data.tensors.v)
    log_likelihood(student, data.tensors.v) |> mean
    randn!(student.weights)
    student.weights .*= 1/sqrt(length(student.vis))
    log_pseudolikelihood_rand(student, data.tensors.v)
    log_likelihood(student, data.tensors.v) |> mean
    train!(student, data; iters = 50000, opt = ADAM(0.001, (0.9, 0.999)))
    log_pseudolikelihood_rand(student, data.tensors.v)
    log_likelihood(student, data.tensors.v) |> mean

    @test cor(free_energy(teacher, data.tensors.v),
              free_energy(student, data.tensors.v)) ≥ 0.8

end

@testset "ReLU sample_h_from_v gradient" begin
    rbm = RBM(Binary(100), ReLU(50))
    randn!(rbm.weights)
    rbm.weights ./= sqrt(length(rbm.vis))
    randn!(rbm.vis.θ)
    randn!(rbm.hid.θ)
    rand!(rbm.hid.γ)
    ps = params(rbm)
    v = sample_v_from_v(rbm, zeros(size(rbm.vis)..., 100); steps=10)
    gs = gradient(ps) do
        h = sample_h_from_v(rbm, v)
        mean(h)
    end
    @test isnothing(gs[rbm.vis.θ])
    @test gs[rbm.weights] ≈ vec(mean(v; dims=2)) * gs[rbm.hid.θ]'
    gs[rbm.weights]
    vec(mean(v; dims=2)) * gs[rbm.hid.θ]'
end