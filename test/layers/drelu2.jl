using Test, Statistics, Random, LinearAlgebra
using Zygote, SpecialFunctions, Flux, Distributions, FiniteDifferences, Juno,
    ValueHistories
using RestrictedBoltzmannMachines
using Flux: params
using Base: front, tail

Random.seed!(579)

@testset "dReLU2 / dReLU convert" begin
    layer = dReLU2(randn(10), randn(10), rand(10), rand(10) .- 0.5)
    layer_ = dReLU2(dReLU(layer))
    @test layer.θ ≈ layer_.θ
    @test layer.Δ ≈ layer_.Δ
    @test layer.γ ≈ layer_.γ
    @test layer.η ≈ layer_.η

    layer = dReLU(randn(10), randn(10), rand(10), rand(10))
    layer_ = dReLU(dReLU2(layer))
    @test layer.θp ≈ layer_.θp
    @test layer.θn ≈ layer_.θn
    @test layer.γp ≈ layer_.γp
    @test layer.γn ≈ layer_.γn
end

@testset "statistics" begin
    layer = dReLU2(randn(10), randn(10), rand(10), rand(10) .- 0.5)
    sample = random(layer, zeros(size(layer)..., 100000))
    @test transfer_mean(layer) ≈ mean(sample; dims=2) rtol=0.1
    @test transfer_std(layer) ≈ std(sample; dims=2) rtol=0.1
    @test transfer_var(layer) ≈ var(sample; dims=2) rtol=0.1
end

@testset "dReLU2, contrastive divergence gradient" begin
    rbm = RBM(Potts(5,10), dReLU2(randn(10,5), randn(10,5), rand(10,5), rand(10,5) .- 0.5))
    randn!(rbm.weights); randn!(rbm.vis.θ)
    ps = Flux.params(rbm)
    @test rbm.weights ∈ ps
    @test rbm.vis.θ ∈ ps
    @test rbm.hid.θ ∈ ps
    @test rbm.hid.Δ ∈ ps
    @test rbm.hid.γ ∈ ps
    @test rbm.hid.η ∈ ps
    vd = random(rbm.vis, zeros(size(rbm.vis, 10)))
    vm = random(rbm.vis, zeros(size(rbm.vis, 10)))
    wd = rand(10)
    gs = gradient(ps) do
        contrastive_divergence(rbm, vd, vm, wd)
    end
    pw = randn(size(rbm.weights)); pg = randn(size(rbm.vis));
    pθ = randn(size(rbm.hid)); pΔ = randn(size(rbm.hid));
    pγ = randn(size(rbm.hid)); pη = randn(size(rbm.hid));
    Δ = central_fdm(5,1)(0) do ϵ
        rbm_ = RBM(Potts(rbm.vis.θ .+ ϵ .* pg),
                   dReLU2(rbm.hid.θ .+ ϵ .* pθ, rbm.hid.Δ .+ ϵ .* pΔ,
                          rbm.hid.γ .+ ϵ .* pγ, rbm.hid.η .+ ϵ .* pη),
                   rbm.weights .+ ϵ .* pw)
        contrastive_divergence(rbm_, vd, vm, wd)
    end
    @test Δ ≈ dot(gs[rbm.weights], pw) + dot(gs[rbm.vis.θ], pg) +
              dot(gs[rbm.hid.θ], pθ) + dot(gs[rbm.hid.Δ], pΔ) +
              dot(gs[rbm.hid.γ], pγ) + dot(gs[rbm.hid.η], pη)
end

@testset "dReLU2 cd training" begin
    teacher = RBM(Binary(10), dReLU2(3))
    randn!(teacher.weights)
    teacher.weights .*= 2/sqrt(length(teacher.vis))
    v = zeros(size(teacher.vis)..., 10000)
    Juno.@progress for t = 1:20
        v .= sample_v_from_v(teacher, v)
    end

    data = Data((v = v, w = ones(size(v)[end])), batchsize=64)
    log_pseudolikelihood_rand(teacher, data.tensors.v)
    log_likelihood(teacher, data.tensors.v) |> mean

    student = RBM(Binary(size(teacher.vis)...), dReLU2(size(teacher.hid)...))
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
    train!(student, data; iters = 100000, opt = ADAM(0.0001, (0.5, 0.999)),
        λw=1e-6, λh=1e-6, λg=1e-6)
    log_pseudolikelihood_rand(student, data.tensors.v)
    log_likelihood(student, data.tensors.v) |> mean

    ρ = cor(free_energy(teacher, data.tensors.v),
            free_energy(student, data.tensors.v))
    @show ρ
    @test ρ ≥ 0.7
end
