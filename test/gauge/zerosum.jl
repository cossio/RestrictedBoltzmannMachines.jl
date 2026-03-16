using Test: @test, @testset
using LinearAlgebra: norm
using Statistics: mean
using RestrictedBoltzmannMachines: zerosum, zerosum!, zerosum_weights, free_energy,
    RBM, Potts, Binary, Spin, Gaussian, ReLU, dReLU, pReLU, xReLU, sample_from_inputs

@testset "zerosum (visible Potts)" begin
    N = (3, 2, 3)
    M = (2, 3)
    rbm = RBM(Potts(; θ = randn(N...)), Binary(; θ = randn(M...)), randn(N..., M...))
    rbm1 = zerosum(rbm)
    @test norm(mean(rbm1.w; dims=1)) < 1e-13
    @test norm(mean(rbm1.visible.θ; dims=1)) < 1e-13
    v = sample_from_inputs(rbm.visible, zeros(N..., 1000))
    F0 = free_energy(rbm, v)
    F1 = free_energy(rbm1, v)
    @test all(F0 - F1 .≈ mean(F0 - F1))
    @test zerosum_weights(rbm.w, rbm) ≈ zerosum(rbm).w
    zerosum!(rbm)
    @test free_energy(rbm, v) ≈ F1
end

@testset "zerosum (hidden Potts)" begin
    N = (2, 3)
    M = (3, 2, 3)
    rbm = RBM(Binary(; θ = randn(N...)), Potts(; θ = randn(M...)), randn(N..., M...))
    rbm1 = zerosum(rbm)
    @test norm(mean(rbm1.w; dims=3)) < 1e-13
    @test norm(mean(rbm1.hidden.θ; dims=1)) < 1e-13
    v = sample_from_inputs(rbm.visible, zeros(N..., 1000))
    F0 = free_energy(rbm, v)
    F1 = free_energy(rbm1, v)
    @test all(F0 - F1 .≈ mean(F0 - F1))
    @test zerosum_weights(rbm.w, rbm) ≈ zerosum(rbm).w
    zerosum!(rbm)
    @test free_energy(rbm, v) ≈ F1
end

@testset "zerosum_weights with different weights array" begin
    # Test that zerosum_weights operates on `weights`, not `rbm.w`
    for (vis, hid, N, M) in [
        (Binary, Potts, (2, 3), (3, 2, 3)),
        (Potts, Binary, (3, 2, 3), (2, 3)),
        (Potts, Potts, (3, 2, 3), (3, 2, 3)),
    ]
        rbm = RBM(vis(; θ = randn(N...)), hid(; θ = randn(M...)), randn(N..., M...))
        w_other = randn(size(rbm.w))
        result = zerosum_weights(w_other, rbm)
        # result should be derived from w_other, not rbm.w
        @test size(result) == size(w_other)
        # applying zerosum_weights twice should be idempotent
        @test zerosum_weights(result, rbm) ≈ result
        # result should NOT equal zerosum_weights applied to rbm.w (different input)
        @test !(result ≈ zerosum_weights(rbm.w, rbm))
    end
end

@testset "zerosum (visible and hidden Potts)" begin
    N = (3, 2, 3)
    M = (3, 2, 3)
    rbm = RBM(Potts(; θ = randn(N...)), Potts(; θ = randn(M...)), randn(N..., M...))
    rbm1 = zerosum(rbm)
    @test norm(mean(rbm1.w; dims=1)) < 1e-13
    @test norm(mean(rbm1.w; dims=4)) < 1e-13
    @test norm(mean(rbm1.visible.θ; dims=1)) < 1e-13
    @test norm(mean(rbm1.hidden.θ; dims=1)) < 1e-13
    v = sample_from_inputs(rbm.visible, zeros(N..., 1000))
    F0 = free_energy(rbm, v)
    F1 = free_energy(rbm1, v)
    @test all(F0 - F1 .≈ mean(F0 - F1))
    @test zerosum_weights(rbm.w, rbm) ≈ zerosum(rbm).w
    zerosum!(rbm)
    @test free_energy(rbm, v) ≈ F1
end

@testset "zerosum (visible Potts, various hidden layers)" begin
    N = (3, 2, 3)
    M = (2, 3)
    hidden_layers = (
        Binary(M), Spin(M), Gaussian(M), ReLU(M), dReLU(M), pReLU(M), xReLU(M)
    )
    for hidden in hidden_layers
        rbm = RBM(Potts(; θ = randn(N...)), hidden, randn(N..., M...))
        rbm1 = zerosum(rbm)
        @test norm(mean(rbm1.w; dims=1)) < 1e-13
        @test norm(mean(rbm1.visible.θ; dims=1)) < 1e-13
        v = v = sample_from_inputs(rbm.visible, zeros(N..., 1000))
        F0 = free_energy(rbm, v)
        F1 = free_energy(rbm1, v)
        @test all(F0 - F1 .≈ mean(F0 - F1))
        zerosum!(rbm)
        @test free_energy(rbm, v) ≈ F1
    end
end
