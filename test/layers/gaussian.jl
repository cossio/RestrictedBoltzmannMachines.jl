using Test, Statistics, Random, LinearAlgebra
using Zygote, Flux, SpecialFunctions, FiniteDifferences
using RestrictedBoltzmannMachines
using Base: front, tail
using RestrictedBoltzmannMachines
using RestrictedBoltzmannMachines: randn_like,  gauss_logpdf, gauss_logcdf, gauss_standardize, __transfer_logcdf, __transfer_logpdf

include("../test_utils.jl")

Random.seed!(569)

layer = Gaussian(randn(10,5), rand(10,5))
ps = params(layer)
gs = gradient(ps) do
    cgf(layer, 1, 2)
end
@test gs[layer.θ] ≈ transfer_mean(layer, 1, 2)
@test 2gs[layer.γ] ≈ -transfer_var(layer, 1, 2) - transfer_mean(layer, 1, 2).^2

gs = gradient(ps) do
    mean([mean(random(layer, 1, 2)) for _ = 1:10^4])
end
gs_ = gradient(params(layer)) do
    mean(transfer_mean(layer, 1, 2))
end
@test gs[layer.θ] ≈ gs_[layer.θ] # this one is exact because the mapping is linear
@test gs[layer.γ] ≈ gs_[layer.γ] rtol=0.1

sample = random(layer, zeros(size(layer)..., 10000))
@test transfer_mean(layer) ≈ mean(sample; dims=3) rtol=0.1
@test transfer_std(layer) ≈ std(sample; dims=3) rtol=0.1
@test transfer_var(layer) ≈ var(sample; dims=3) rtol=0.1
@test transfer_mean_abs(layer) ≈ mean(abs, sample; dims=3) rtol=0.1
@test iszero(gauss_standardize.(layer.θ, layer.γ, transfer_mean(layer)))
@test gauss_standardize.(layer.θ, layer.γ, 0) ≈ -transfer_mean(layer) ./ transfer_std(layer)

@testset "Gaussian pdf, cdf" begin
    for _ = 1:10
        θ, γ, x = randn(3)
        @test exp(gauss_logpdf(θ, γ, x)) ≈ only(gradient(x -> exp(gauss_logcdf(θ, γ, x)), x))
    end
end

@testset "Gaussian energy & cgf gradients" begin
    θ, γ = randn(2,3), rand(2,3)
    # with batch dimensions
    x = rand(2,3, 1,2)
    (dI,) = gradient(I -> sum(cgf(Gaussian(θ, γ), I)), x)
    @test dI ≈ transfer_mean(Gaussian(θ, γ), x)
    gradtest((θ, γ) -> energy(Gaussian(θ, γ), x), θ, γ)
    gradtest((θ, γ) -> cgf(Gaussian(θ, γ), x), θ, γ)
    
    # without batch dimensions
    x = randn(2,3)
    (dI,) = gradient(I -> cgf(Gaussian(θ, γ), I), x)
    @test dI ≈ transfer_mean(Gaussian(θ, γ), x)
    gradtest((θ, γ) -> energy(Gaussian(θ, γ), x), θ, γ)
    gradtest((θ, γ) -> cgf(Gaussian(θ, γ), x), θ, γ)
end

@testset "Gaussian random gradient" begin
    layer = Gaussian(randn(5,5), randn(5,5))
    ps = Flux.params(layer)
    h = random(layer)
    gs = gradient(ps) do
        Zygote.@ignore Random.seed!(1)
        h_ = random(layer)
        Zygote.@ignore h .= h_
        sum(h_)
    end
    pdf = exp.(__transfer_logpdf(layer, h))
    ∇cdf = gradient(ps) do
        sum(exp.(__transfer_logcdf(layer, h)))
    end
    @test gs[layer.θ] ≈ -∇cdf[layer.θ] ./ pdf
    @test gs[layer.γ] ≈ -∇cdf[layer.γ] ./ pdf
end

@testset "Gaussian, contrastive divergence gradient" begin
    rbm = RBM(Potts(5,10), Gaussian(randn(10,5), rand(10,5)))
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
        contrastive_divergence_v(rbm, vd, vm, wd)
    end
    pw = randn(size(rbm.weights)); pg = randn(size(rbm.vis));
    pθ = randn(size(rbm.hid)); pγ = randn(size(rbm.hid));
    Δ = central_fdm(5,1)(0) do ϵ
        rbm_ = RBM(Potts(rbm.vis.θ .+ ϵ .* pg),
                   Gaussian(rbm.hid.θ .+ ϵ .* pθ, rbm.hid.γ .+ ϵ .* pγ),
                   rbm.weights .+ ϵ .* pw)
        contrastive_divergence_v(rbm_, vd, vm, wd)
    end
    @test Δ ≈ dot(gs[rbm.weights], pw) + dot(gs[rbm.vis.θ], pg) +
              dot(gs[rbm.hid.θ], pθ) + dot(gs[rbm.hid.γ], pγ)
end

@testset "Gaussian sample_h_from_v gradient" begin
    rbm = RBM(Binary(100), Gaussian(100))
    randn!(rbm.weights)
    rbm.weights ./= sqrt(length(rbm.vis))
    randn!(rbm.vis.θ)
    randn!(rbm.hid.θ)
    randn!(rbm.hid.γ)
    ps = params(rbm)
    v = sample_v_from_v(rbm, zeros(size(rbm.vis)...); steps=10)
    h = sample_h_from_v(rbm, v)
    gs = gradient(ps) do
        h_ = sample_h_from_v(rbm, v)
        Zygote.@ignore h .= h_
        mean(2 .* h_ .+ 1)
    end
    @test isnothing(gs[rbm.vis.θ])
    @test gs[rbm.weights] ≈ v * gs[rbm.hid.θ]'

    pdf = exp.(__transfer_logpdf(rbm.hid, h, inputs_v_to_h(rbm, v)))
    gs_ = gradient(ps) do
        cdf = exp.(__transfer_logcdf(rbm.hid, h, inputs_v_to_h(rbm, v)))
        2mean(-cdf ./ pdf) + 1
    end
    @test isnothing(gs_[rbm.vis.θ])
    @test gs[rbm.hid.θ] ≈ gs_[rbm.hid.θ]
    @test gs[rbm.hid.γ] ≈ gs_[rbm.hid.γ]
    @test gs[rbm.weights] ≈ gs_[rbm.weights]
end
