using Test, Statistics, Random, LinearAlgebra
using Zygote, Flux, SpecialFunctions, FiniteDifferences
using RestrictedBoltzmannMachines
using Base: front, tail
using RestrictedBoltzmannMachines

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

@testset "Gaussian energy & cgf gradients" begin
    # with batch dimensions
    θ = randn(4,5,6)
    γ = randn(4,5,6)
    x = rand(Bool, 4,5,6, 3,2)
    pθ, pγ = randn(size(θ)), randn(size(γ))
    testfun(θ, γ) = sum(sin.(energy(Gaussian(θ, γ), x)))
    dθ, dγ = gradient(testfun, θ, γ)
    @test central_fdm(5,1)(ϵ -> testfun(θ + ϵ * pθ, γ + ϵ * pγ), 0) ≈ sum(dθ .* pθ) + sum(dγ .* pγ)
    testfun(θ, γ) = sum(sin.(cgf(Gaussian(θ, γ), x)))
    dθ, dγ = gradient(testfun, θ, γ)
    @test central_fdm(5,1)(ϵ -> testfun(θ + ϵ * pθ, γ + ϵ * pγ), 0) ≈ sum(dθ .* pθ) + sum(dγ .* pγ)

    # energy without batch dimensions
    Random.seed!(2)
    θ = randn(4,5,6)
    γ = randn(4,5,6)
    x = rand(Bool, 4,5,6)
    pθ, pγ = randn(size(θ)), randn(size(γ))
    testfun(θ, γ) = sum(sin.(energy(Gaussian(θ, γ), x)))
    dθ, dγ = gradient(testfun, θ, γ)
    @test central_fdm(5,1)(ϵ -> testfun(θ + ϵ * pθ, γ + ϵ * pγ), 0) ≈ sum(dθ .* pθ) + sum(dγ .* pγ)
    testfun(θ, γ) = sum(sin.(cgf(Gaussian(θ, γ), x)))
    dθ, dγ = gradient(testfun, θ, γ)
    @test central_fdm(5,1)(ϵ -> testfun(θ + ϵ * pθ, γ + ϵ * pγ), 0) ≈ sum(dθ .* pθ) + sum(dγ .* pγ)
end

@testset "Gaussian random gradient" begin
    layer = Gaussian(randn(10,5), rand(10,5))
    ps = params(layer)
    gs = gradient(ps) do
        Zygote.@ignore Random.seed!(1)
        mean(random(layer))
    end
    pθ, pγ = randn(size(layer)), randn(size(layer))
    Δ = central_fdm(5,1)(0) do ϵ
        Random.seed!(1)
        layer_ = Gaussian(layer.θ + ϵ * pθ, layer.γ + ϵ * pγ)
        mean(random(layer_))
    end
    @test Δ ≈ dot(gs[layer.θ], pθ) + dot(gs[layer.γ], pγ)
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
        contrastive_divergence(rbm, vd, vm, wd)
    end
    pw = randn(size(rbm.weights)); pg = randn(size(rbm.vis));
    pθ = randn(size(rbm.hid)); pγ = randn(size(rbm.hid));
    Δ = central_fdm(5,1)(0) do ϵ
        rbm_ = RBM(Potts(rbm.vis.θ .+ ϵ .* pg),
                   Gaussian(rbm.hid.θ .+ ϵ .* pθ, rbm.hid.γ .+ ϵ .* pγ),
                   rbm.weights .+ ϵ .* pw)
        contrastive_divergence(rbm_, vd, vm, wd)
    end
    @test Δ ≈ dot(gs[rbm.weights], pw) + dot(gs[rbm.vis.θ], pg) +
              dot(gs[rbm.hid.θ], pθ) + dot(gs[rbm.hid.γ], pγ)
end
