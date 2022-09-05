using Test: @test, @testset
using LinearAlgebra: norm
using Statistics: mean
using RestrictedBoltzmannMachines: zerosum, zerosum!, zerosum_weights, extensive_sample, free_energy,
    RBM, Potts, Binary, Spin, Gaussian, ReLU, dReLU, pReLU, xReLU

@testset "zerosum (visible Potts)" begin
    q = 3
    N = (2, 3)
    M = (2, 3)
    rbm = RBM(Potts(; θ = randn(q, N...)), Binary(; θ = randn(M...)), randn(q, N..., M...))
    rbm1 = zerosum(rbm)
    @test norm(mean(rbm1.w; dims=1)) < 1e-13
    @test norm(mean(rbm1.visible.θ; dims=1)) < 1e-13
    v = extensive_sample(rbm.visible)
    F0 = free_energy(rbm, v)
    F1 = free_energy(rbm1, v)
    @test all(F0 - F1 .≈ mean(F0 - F1))
    @test zerosum_weights(rbm.w, rbm) ≈ zerosum(rbm).w
    zerosum!(rbm)
    @test free_energy(rbm, v) ≈ F1
end

@testset "zerosum (hidden Potts)" begin
    p = 3
    N = (2, 3)
    M = (2, 3)
    rbm = RBM(Binary(; θ = randn(N...)), Potts(; θ = randn(p, M...)), randn(N..., p, M...))
    rbm1 = zerosum(rbm)
    @test norm(mean(rbm1.w; dims=3)) < 1e-13
    @test norm(mean(rbm1.hidden.θ; dims=1)) < 1e-13
    v = extensive_sample(rbm.visible)
    F0 = free_energy(rbm, v)
    F1 = free_energy(rbm1, v)
    @test all(F0 - F1 .≈ mean(F0 - F1))
    @test zerosum_weights(rbm.w, rbm) ≈ zerosum(rbm).w
    zerosum!(rbm)
    @test free_energy(rbm, v) ≈ F1
end

@testset "zerosum (visible and hidden Potts)" begin
    q = 3
    p = 3
    N = (2, 3)
    M = (2, 3)
    rbm = RBM(Potts(; θ = randn(q, N...)), Potts(; θ = randn(p, M...)), randn(q, N..., p, M...))
    rbm1 = zerosum(rbm)
    @test norm(mean(rbm1.w; dims=1)) < 1e-13
    @test norm(mean(rbm1.w; dims=4)) < 1e-13
    @test norm(mean(rbm1.visible.θ; dims=1)) < 1e-13
    @test norm(mean(rbm1.hidden.θ; dims=1)) < 1e-13
    v = extensive_sample(rbm.visible)
    F0 = free_energy(rbm, v)
    F1 = free_energy(rbm1, v)
    @test all(F0 - F1 .≈ mean(F0 - F1))
    @test zerosum_weights(rbm.w, rbm) ≈ zerosum(rbm).w
    zerosum!(rbm)
    @test free_energy(rbm, v) ≈ F1
end

@testset "zerosum (visible Potts, various hidden layers)" begin
    q = 3
    N = (2, 3)
    M = (2, 3)
    hidden_layers = (
        Binary(M), Spin(M), Gaussian(M), ReLU(M), dReLU(M), pReLU(M), xReLU(M)
    )
    for hidden in hidden_layers
        rbm = RBM(Potts(; θ = randn(q, N...)), hidden, randn(q, N..., M...))
        rbm1 = zerosum(rbm)
        @test norm(mean(rbm1.w; dims=1)) < 1e-13
        @test norm(mean(rbm1.visible.θ; dims=1)) < 1e-13
        v = extensive_sample(rbm.visible)
        F0 = free_energy(rbm, v)
        F1 = free_energy(rbm1, v)
        @test all(F0 - F1 .≈ mean(F0 - F1))
        zerosum!(rbm)
        @test free_energy(rbm, v) ≈ F1
    end
end
