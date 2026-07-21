import Zygote
import RestrictedBoltzmannMachines as RBMs
using RestrictedBoltzmannMachines: RBM, Potts, Binary, Gaussian
using RestrictedBoltzmannMachines: sample_from_inputs, zerosum!, free_energy
using RestrictedBoltzmannMachines.Experimental.Sparse: proxpcd!,
    prox_glasso!, prox_gl2l1!, regularization_penalty,
    ∂glasso_weights, ∂gl2l1_weights, _prox_squared_l1_threshold
using LinearAlgebra: norm
using Random: seed!
using Statistics: mean
using Test: @test, @testset, @test_throws

@testset "group-lasso subgradient matches autodiff" begin
    # Potts visible + Potts hidden: glasso groups over both color axes, gl2l1 over visible.
    rbm = RBM(Potts(; θ = randn(3, 4)), Potts(; θ = randn(2, 3)), randn(3, 4, 2, 3))
    glasso_weights, gl2l1_weights = rand(2)
    gs = Zygote.gradient(rbm) do rbm
        regularization_penalty(rbm; glasso_weights, gl2l1_weights)
    end
    ∂ = ∂glasso_weights(rbm.w, glasso_weights, rbm) .+ ∂gl2l1_weights(rbm.w, gl2l1_weights, rbm.visible)
    @test only(gs).w ≈ ∂
end

@testset "penalty is NaN-safe at zero groups" begin
    rbm = RBM(Potts(; θ = randn(3, 4)), Binary(; θ = randn(2)), randn(3, 4, 2))
    rbm.w[:, 1, 1] .= 0
    glasso_weights, gl2l1_weights = rand(2)
    gs = only(Zygote.gradient(rbm -> regularization_penalty(rbm; glasso_weights, gl2l1_weights), rbm))
    @test all(isfinite, gs.w)
    @test all(iszero, gs.w[:, 1, 1])
end

@testset "prox_glasso! block soft-threshold and exact zeros" begin
    rbm = RBM(Potts(; θ = randn(3, 4)), Binary(; θ = randn(2)), randn(3, 4, 2))
    zerosum!(rbm)
    w0 = copy(rbm.w)
    t = 0.3
    prox_glasso!(rbm, t)
    for i in 1:4, μ in 1:2
        g = norm(w0[:, i, μ])
        @test rbm.w[:, i, μ] ≈ max(0, 1 - t / g) * w0[:, i, μ]
        g ≤ t && @test all(iszero, rbm.w[:, i, μ])
    end
    @test all(abs.(sum(rbm.w; dims = 1)) .< 1.0e-10) # preserves zerosum
end

@testset "prox_glasso! zeros survive zerosum! for Potts hidden" begin
    rbm = RBM(Potts(; θ = randn(3, 4)), Potts(; θ = randn(2, 5)), randn(3, 4, 2, 5))
    zerosum!(rbm)
    g = vec(sqrt.(sum(abs2, rbm.w; dims = (1, 3))))
    t = (sort(g)[1] + sort(g)[2]) / 2
    prox_glasso!(rbm, t)
    zeroed = vec(iszero.(sqrt.(sum(abs2, rbm.w; dims = (1, 3)))))
    @test any(zeroed)
    zerosum!(rbm)
    @test vec(iszero.(sqrt.(sum(abs2, rbm.w; dims = (1, 3))))) == zeroed
    @test all(abs.(sum(rbm.w; dims = 1)) .< 1.0e-10)
    @test all(abs.(sum(rbm.w; dims = 3)) .< 1.0e-10)
end

@testset "_prox_squared_l1_threshold self-consistency" begin
    # τ must satisfy the fixed point τ = c·∑ᵢ max(0, aᵢ − τ).
    for _ in 1:20
        a = abs.(randn(8)) .+ 0.01
        c = 0.2 + rand()
        τ = _prox_squared_l1_threshold(a, c)
        @test τ ≈ c * sum(max.(0, a .- τ))
        @test τ ≥ 0
    end
end

@testset "prox_gl2l1! reduces to squared-ℓ1 prox for Binary" begin
    rbm = RBM(Binary(; θ = randn(5)), Binary(; θ = randn(3)), randn(5, 3))
    w0 = copy(rbm.w)
    t = 1.0
    prox_gl2l1!(rbm, t)
    for μ in 1:3
        a = abs.(w0[:, μ])
        τ = _prox_squared_l1_threshold(a, t / 5) # N = 5 sites
        @test rbm.w[:, μ] ≈ sign.(w0[:, μ]) .* max.(0, a .- τ)
    end
end

@testset "prox_gl2l1! exact zeros and gauge preservation (Potts)" begin
    rbm = RBM(Potts(; θ = randn(3, 6)), Binary(; θ = randn(4)), randn(3, 6, 4))
    zerosum!(rbm)
    prox_gl2l1!(rbm, 2.0)
    en = vec(sqrt.(sum(abs2, rbm.w; dims = 1)))
    @test any(iszero, en)                                  # exact-zero edges
    @test all(isfinite, rbm.w)
    @test all(abs.(sum(rbm.w; dims = 1)) .< 1.0e-10)       # preserves zerosum
end

@testset "proxpcd! glasso: subgradient (prox=false) and proximal (prox=true)" begin
    seed!(0)
    data = sample_from_inputs(Potts((3, 6)), zeros(3, 6, 200))

    # prox = false: plain subgradient training, must run finite
    rbm = RBM(Potts(; θ = randn(3, 6)), Gaussian(; θ = randn(4), γ = ones(4)), 0.1 * randn(3, 6, 4))
    proxpcd!(rbm, data; iters = 20, batchsize = 32, glasso_weights = 0.05, prox = false)
    @test all(isfinite, rbm.w)

    # prox = true: exact zeros, model ends in zerosum gauge
    rbm = RBM(Potts(; θ = randn(3, 6)), Gaussian(; θ = randn(4), γ = ones(4)), 0.1 * randn(3, 6, 4))
    proxpcd!(rbm, data; iters = 40, batchsize = 32, glasso_weights = 0.5, prox = true, rescale = true)
    @test all(isfinite, rbm.w)
    @test maximum(abs, sum(rbm.w; dims = 1)) < 1.0e-8
    @test any(iszero, vec(sqrt.(sum(abs2, rbm.w; dims = 1))))
end

@testset "proxpcd! gl2l1: subgradient (prox=false) and proximal (prox=true)" begin
    seed!(0)
    data = sample_from_inputs(Potts((3, 6)), zeros(3, 6, 200))

    rbm = RBM(Potts(; θ = randn(3, 6)), Binary(; θ = randn(4)), 0.1 * randn(3, 6, 4))
    proxpcd!(rbm, data; iters = 20, batchsize = 32, gl2l1_weights = 0.05, prox = false)
    @test all(isfinite, rbm.w)

    rbm = RBM(Potts(; θ = randn(3, 6)), Binary(; θ = randn(4)), 0.1 * randn(3, 6, 4))
    proxpcd!(rbm, data; iters = 40, batchsize = 32, gl2l1_weights = 0.5, prox = true, rescale = false)
    @test all(isfinite, rbm.w)
    @test any(iszero, vec(sqrt.(sum(abs2, rbm.w; dims = 1))))
end

@testset "proxpcd! enforces at most one group-lasso penalty" begin
    data = sample_from_inputs(Potts((3, 4)), zeros(3, 4, 32))
    mk() = RBM(Potts(; θ = randn(3, 4)), Binary(; θ = randn(2)), 0.1 * randn(3, 4, 2))
    @test_throws AssertionError proxpcd!(mk(), data; iters = 1, gl2l1_weights = 0.1, glasso_weights = 0.1)
    # each alone, and combined with the smooth l2_weights, is fine
    proxpcd!(mk(), data; iters = 1, glasso_weights = 0.1, l2_weights = 0.1)
    proxpcd!(mk(), data; iters = 1, gl2l1_weights = 0.1)
    @test true
end
