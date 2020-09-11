using Test, Statistics, Random, LinearAlgebra
using Zygote, SpecialFunctions, Flux, Distributions, FiniteDifferences, ProgressMeter
using RestrictedBoltzmannMachines
using Flux: params
using Base: front, tail
using RestrictedBoltzmannMachines: ∇relu_rand, ∇relu_cgf, ∇drelu_rand, mills,
    drelu_rand, drelu_survival, drelu_pdf

include("../test_utils.jl")

Random.seed!(569)

(θp, θn, γp, γn) = (randn(), randn(), rand(), rand())

@testset "statistics" begin
    layer = dReLU(randn(10), randn(10), rand(10), rand(10))
    sample = random(layer, zeros(size(layer)..., 100000))
    @test transfer_mean(layer) ≈ mean(sample; dims=2) rtol=0.1
    @test transfer_std(layer) ≈ std(sample; dims=2) rtol=0.1
    @test transfer_var(layer) ≈ var(sample; dims=2) rtol=0.1
end

@testset "dReLU, cgf and moments" begin
    dθp, dθn, dγp, dγn = gradient(5.0, -2.0, 1.0, 3.0) do θp, θn, γp, γn
        cgf(dReLU([θp], [θn], [γp], [γn]))
    end
    pp, pn = first.(RBMs.probs_pair(dReLU([5.0], [-2.0], [1.0], [3.0])))
    @test first(pp .+ pn) ≈ 1
    @test dθp ≈  first(transfer_mean(ReLU([5.0], [1.0]))) * pp
    @test dθn ≈ -first(transfer_mean(ReLU([2.0], [3.0]))) * pn
    @test -2dγp ≈ first(transfer_var(ReLU([5.0], [1.0]))) * pp + dθp^2 / pp
    @test -2dγn ≈ first(transfer_var(ReLU([2.0], [3.0]))) * pn + dθn^2 / pn
end

@testset "dReLU, survival" begin
    x, dθp_, dθn_, dγp_, dγn_ = ∇drelu_rand(θp, θn, γp, γn)
    dθp, dθn, dγp, dγn, dx = gradient(drelu_survival, θp, θn, γp, γn, x)
    @test drelu_pdf(θp, θn, γp, γn, x) ≈ -dx
    @test dθp_ ≈ -dθp / dx
    @test dθn_ ≈ -dθn / dx
    @test dγp_ ≈ -dγp / dx
    @test dγn_ ≈ -dγn / dx
end

dθp, dθn, dγp, dγn  = gradient(θp, θn, γp, γn) do θp, θn, γp, γn
    mean(transfer_mean(dReLU([θp], [θn], [γp], [γn])))
end
dθp_, dθn_, dγp_, dγn_ = gradient(θp, θn, γp, γn) do θp, θn, γp, γn
    mean([drelu_rand(θp, θn, γp, γn) for _ = 1:10^4])
end
@test dθp_ ≈ dθp atol = 0.1
@test dθn_ ≈ dθn atol = 0.1
@test dγp_ ≈ dγp atol = 0.1
@test dγn_ ≈ dγn atol = 0.1

dθp, dθn, dγp, dγn  = gradient(θp, θn, γp, γn) do θp, θn, γp, γn
    mean(transfer_std(dReLU([θp], [θn], [γp], [γn])))
end
dθp_, dθn_, dγp_, dγn_ = gradient(θp, θn, γp, γn) do θp, θn, γp, γn
    std([drelu_rand(θp, θn, γp, γn) for _ = 1:30000])
end
@test dθp_ ≈ dθp atol = 0.1
@test dθn_ ≈ dθn atol = 0.1
@test dγp_ ≈ dγp atol = 0.1
@test dγn_ ≈ dγn atol = 0.1

dθp, dθn, dγp, dγn = gradient(5.0, -2.0, 1.0, 3.0) do θp, θn, γp, γn
    RBMs.drelu_cgf(θp, θn, γp, γn)
end
pp, pn = first.(RBMs.probs_pair(dReLU([5.0], [-2.0], [1.0], [3.0])))
@test first(pp .+ pn) ≈ 1
@test dθp ≈  first(transfer_mean(ReLU([5.0], [1.0]))) * pp
@test dθn ≈ -first(transfer_mean(ReLU([2.0], [3.0]))) * pn
@test -2dγp ≈ first(transfer_var(ReLU([5.0], [1.0]))) * pp + dθp^2 / pp
@test -2dγn ≈ first(transfer_var(ReLU([2.0], [3.0]))) * pn + dθn^2 / pn

layer = dReLU(randn(10), randn(10), rand(10), rand(10))
ps = params(layer)
gs = gradient(ps) do
    cgf(layer)
end
@test   gs[layer.θp] + gs[layer.θn]  ≈ transfer_mean(layer)
@test -2gs[layer.γp] - 2gs[layer.γn] ≈ transfer_var(layer) .+ transfer_mean(layer).^2

pp, pn = RBMs.probs_pair(layer)
@test pp .+ pn ≈ ones(10)
ds = [MixtureModel([
        truncated(Normal(layer.θp[i]/layer.γp[i], √(1/layer.γp[i])), 0, +Inf),
        truncated(Normal(layer.θn[i]/layer.γn[i], √(1/layer.γn[i])), -Inf, 0)
    ],
    [pp[i], pn[i]]) for i = 1:10]
@test mean.(ds) ≈ transfer_mean(layer)
@test std.(ds) ≈ transfer_std(layer)
@test var.(ds) ≈ transfer_var(layer)

layer = dReLU(randn(10), randn(10), rand(10), rand(10))
pp, pn = RBMs.probs_pair(layer)
sample = random(layer, zeros(size(layer)..., 100000))
@test mean(sample; dims=2) ≈ transfer_mean(layer) rtol=0.01
@test std(sample; dims=2) ≈ transfer_std(layer) rtol=0.01
@test var(sample; dims=2) ≈ transfer_var(layer) rtol=0.01
@test mean(sample .> 0; dims=2) ≈ pp rtol=0.01
@test mean(sample .< 0; dims=2) ≈ pn rtol=0.01

layer = dReLU(randn(10,5), randn(10,5), rand(10,5), rand(10,5))
sample = random(layer, zeros(size(layer)..., 100000))
@test transfer_mean(layer) ≈ mean(sample; dims=3) rtol=0.01
@test transfer_std(layer) ≈ std(sample; dims=3) rtol=0.01
@test transfer_var(layer) ≈ var(sample; dims=3) rtol=0.01
@test transfer_mean_abs(layer) ≈ mean(abs, sample; dims=3) rtol=0.01

@testset "dReLU energy & cgf gradients" begin
    θp, γp = randn(3,2), rand(3,2)
    θn, γn = randn(3,2), rand(3,2)
    # with batch dimensions
    x = randn(3,2, 1,2)
    (dI,) = gradient(I -> sum(cgf(dReLU(θp, θn, γp, γn), I)), x)
    @test dI ≈ transfer_mean(dReLU(θp, θn, γp, γn), x)
    gradtest((θp, θn, γp, γn) -> energy(dReLU(θp, θn, γp, γn), x), θp, θn, γp, γn)
    #gradtest((θp, θn, γp, γn) -> cgf(dReLU(θp, θn, γp, γn), x), θp, θn, γp, γn)
    gradtest(θp -> cgf(dReLU(θp, θn, γp, γn), x), θp)
    gradtest(θn -> cgf(dReLU(θp, θn, γp, γn), x), θn)
    gradtest(γp -> cgf(dReLU(θp, θn, γp, γn), x), γp)
    gradtest(γn -> cgf(dReLU(θp, θn, γp, γn), x), γn)
    # without batch dimensions
    x = randn(3,2)
    (dI,) = gradient(I -> cgf(dReLU(θp, θn, γp, γn), I), x)
    @test dI ≈ transfer_mean(dReLU(θp, θn, γp, γn), x)
    gradtest((θp, θn, γp, γn) -> energy(dReLU(θp, θn, γp, γn), x), θp, θn, γp, γn)
    gradtest((θp, θn, γp, γn) -> cgf(dReLU(θp, θn, γp, γn), x), θp, θn, γp, γn)
end

@testset "dReLU prob pair" begin
    layer = dReLU([-3.0], [2.0], [2.0], [0.5])
    pp, pn = RBMs.probs_pair(layer)
    @test first(pp) ≈ 0.38634597748536126
    @test first(pn) ≈ 0.6136540225146387
end

@testset "dReLU, contrastive divergence gradient" begin
    rbm = RBM(Potts(5,10), dReLU(randn(10,5), randn(10,5), rand(10,5), rand(10,5)))
    randn!(rbm.weights); randn!(rbm.vis.θ);
    randn!(rbm.hid.θp); randn!(rbm.hid.θn); rand!(rbm.hid.γp); rand!(rbm.hid.γn);
    ps = Flux.params(rbm)
    @test rbm.weights ∈ ps
    @test rbm.vis.θ ∈ ps
    @test rbm.hid.θp ∈ ps
    @test rbm.hid.θn ∈ ps
    @test rbm.hid.γp ∈ ps
    @test rbm.hid.γn ∈ ps
    vd = random(rbm.vis, zeros(size(rbm.vis, 10)))
    vm = random(rbm.vis, zeros(size(rbm.vis, 10)))
    wd = rand(10)
    gs = gradient(ps) do
        contrastive_divergence(rbm, vd, vm, wd)
    end
    pw = randn(size(rbm.weights)); pg = randn(size(rbm.vis));
    pθp = randn(size(rbm.hid)); pθn = randn(size(rbm.hid)); pγp = randn(size(rbm.hid)); pγn = randn(size(rbm.hid));
    Δ = central_fdm(5,1)(0) do ϵ
        rbm_ = RBM(Potts(rbm.vis.θ .+ ϵ .* pg),
                   dReLU(rbm.hid.θp .+ ϵ .* pθp, rbm.hid.θn .+ ϵ .* pθn,
                         rbm.hid.γp .+ ϵ .* pγp, rbm.hid.γn .+ ϵ .* pγn),
                   rbm.weights .+ ϵ .* pw)
        contrastive_divergence(rbm_, vd, vm, wd)
    end
    @test Δ ≈ dot(gs[rbm.weights], pw) + dot(gs[rbm.vis.θ], pg) +
              dot(gs[rbm.hid.θp], pθp) + dot(gs[rbm.hid.θn], pθn) +
              dot(gs[rbm.hid.γp], pγp) + dot(gs[rbm.hid.γn], pγn)
end

@testset "dReLU random gradient" begin
    layer = dReLU(randn(2,2), randn(2,2), rand(2,2), rand(2,2))
    ps = Flux.params(layer)
    gs = gradient(ps) do
        Zygote.@ignore Random.seed!(1)
        mean(random(layer, zeros(size(layer)..., 1000000)))
    end
    gs_ = gradient(ps) do
        mean(transfer_mean(layer))
    end
    @test gs[layer.θp] ≈ gs_[layer.θp] rtol=0.01
    @test gs[layer.θn] ≈ gs_[layer.θn] rtol=0.01
    @test gs[layer.γp] ≈ gs_[layer.γp] rtol=0.01
    @test gs[layer.γn] ≈ gs_[layer.γn] rtol=0.01
end

@testset "dReLU vs Gaussian" begin
    drelu = dReLU(20); gauss = Gaussian(20)
    randn!(gauss.θ); rand!(gauss.γ)
    drelu.θp .= drelu.θn .= gauss.θ
    drelu.γp .= drelu.γn .= gauss.γ
    @test transfer_mean(drelu) ≈ transfer_mean(gauss)
    @test transfer_std(drelu) ≈ transfer_std(gauss)
    @test transfer_var(drelu) ≈ transfer_var(gauss)
    @test transfer_mean_abs(drelu) ≈ transfer_mean_abs(gauss)
    @test transfer_mode(drelu) ≈ transfer_mode(gauss)
    @test RBMs.__cgf(drelu) ≈ RBMs.__cgf(gauss)

    pp, pn = RBMs.probs_pair(drelu)
    @test pn ≈ erfc.(transfer_mean(gauss) ./ √2 ./ transfer_std(gauss)) ./ 2
    @test all(pp .+ pn .≈ 1)

    drelu_rbm = RBM(Binary(30), drelu)
    gauss_rbm = RBM(Binary(30), gauss)
    randn!(gauss_rbm.weights)
    drelu_rbm.weights .= gauss_rbm.weights
    v = random(gauss_rbm.vis, zeros(size(gauss_rbm.vis)..., 100))
    @test free_energy(drelu_rbm, v) ≈ free_energy(gauss_rbm, v)
end

@testset "dReLU cd training" begin
    Random.seed!(3)

    teacher = RBM(Binary(7), dReLU(2));
    randn!(teacher.weights)
    teacher.weights .*= 5/sqrt(length(teacher.vis))
    v = zeros(size(teacher.vis)..., 3000)
    @showprogress for t = 1:1000
        h = sample_h_from_v(teacher, v)
        v .= sample_v_from_h(teacher, h)
    end
    data = Data((v = v, w = ones(size(v)[end])), batchsize=64)

    student = RBM(Binary(size(teacher.vis)...), dReLU(size(teacher.hid)...))
    init!(student, data.tensors.v; eps=1e-10, w=1)
    train!(student, data; iters = 100000, opt = ADAM(0.001, (0.9, 0.999)), λw=1e-5, λh=1e-5, λg=1e-5)
    @test cor(free_energy(teacher, data.tensors.v),
              free_energy(student, data.tensors.v)) ≥ 0.8
end
