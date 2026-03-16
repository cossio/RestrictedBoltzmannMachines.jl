using Test: @test, @testset
using LinearAlgebra: norm
using Statistics: mean
using RestrictedBoltzmannMachines: zerosum, zerosum!, zerosum_weights, free_energy,
    RBM, Potts, Binary, Spin, Gaussian, ReLU, dReLU, pReLU, xReLU, sample_from_inputs,
    PottsGumbel, potts_to_gumbel, gumbel_to_potts, ∂RBM, ∂free_energy

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

@testset "zerosum no-op for non-Potts RBM" begin
    # zerosum on a Binary-Binary RBM should return the same RBM unchanged
    N = (2, 3)
    M = (4, 5)
    rbm = RBM(Binary(; θ = randn(N...)), Binary(; θ = randn(M...)), randn(N..., M...))
    rbm1 = zerosum(rbm)
    @test rbm1 === rbm
    # zerosum_weights should also be a no-op
    @test zerosum_weights(rbm.w, rbm) === rbm.w
end

@testset "zerosum! ∂RBM (hidden Potts)" begin
    N = (2, 3)
    M = (3, 2, 3)
    rbm = RBM(Binary(; θ = randn(N...)), Potts(; θ = randn(M...)), randn(N..., M...))
    zerosum!(rbm)
    v = sample_from_inputs(rbm.visible, zeros(N..., 100))
    ∂ = ∂free_energy(rbm, v)
    zerosum!(∂, rbm)
    # gradient on hidden should be zero-sum in dim 1
    @test norm(mean(∂.hidden; dims=1)) < 1e-13
    # gradient on w should be zero-sum in the hidden Potts dim (dim 3 = ndims(visible)+1)
    @test norm(mean(∂.w; dims=ndims(rbm.visible) + 1)) < 1e-13
end

@testset "zerosum PottsGumbel (visible PottsGumbel, hidden Binary)" begin
    N = (3, 2, 3)
    M = (2, 3)
    rbm_g = RBM(PottsGumbel(; θ = randn(N...)), Binary(; θ = randn(M...)), randn(N..., M...))
    rbm_p = gumbel_to_potts(rbm_g)

    # zerosum on PottsGumbel should equal potts_to_gumbel(zerosum(gumbel_to_potts(...)))
    rbm_g_zs = zerosum(rbm_g)
    rbm_p_zs = zerosum(rbm_p)
    @test rbm_g_zs.visible isa PottsGumbel
    @test rbm_g_zs.visible.par ≈ PottsGumbel(rbm_p_zs.visible).par
    @test rbm_g_zs.w ≈ rbm_p_zs.w
    # check free energy equivalence
    v = sample_from_inputs(rbm_g.visible, zeros(N..., 100))
    F0 = free_energy(rbm_g, v)
    F1 = free_energy(rbm_g_zs, v)
    @test all(F0 - F1 .≈ mean(F0 - F1))
end

@testset "zerosum PottsGumbel (hidden PottsGumbel, visible Binary)" begin
    N = (2, 3)
    M = (3, 2, 3)
    rbm_g = RBM(Binary(; θ = randn(N...)), PottsGumbel(; θ = randn(M...)), randn(N..., M...))
    rbm_p = gumbel_to_potts(rbm_g)

    rbm_g_zs = zerosum(rbm_g)
    rbm_p_zs = zerosum(rbm_p)
    @test rbm_g_zs.hidden isa PottsGumbel
    @test rbm_g_zs.hidden.par ≈ PottsGumbel(rbm_p_zs.hidden).par
    @test rbm_g_zs.w ≈ rbm_p_zs.w
    v = sample_from_inputs(rbm_g.visible, zeros(N..., 100))
    F0 = free_energy(rbm_g, v)
    F1 = free_energy(rbm_g_zs, v)
    @test all(F0 - F1 .≈ mean(F0 - F1))
end

@testset "zerosum PottsGumbel (both PottsGumbel)" begin
    N = (3, 2, 3)
    M = (3, 2, 3)
    rbm_g = RBM(PottsGumbel(; θ = randn(N...)), PottsGumbel(; θ = randn(M...)), randn(N..., M...))
    rbm_p = gumbel_to_potts(rbm_g)

    rbm_g_zs = zerosum(rbm_g)
    rbm_p_zs = zerosum(rbm_p)
    @test rbm_g_zs.visible isa PottsGumbel
    @test rbm_g_zs.hidden isa PottsGumbel
    @test rbm_g_zs.visible.par ≈ PottsGumbel(rbm_p_zs.visible).par
    @test rbm_g_zs.hidden.par ≈ PottsGumbel(rbm_p_zs.hidden).par
    @test rbm_g_zs.w ≈ rbm_p_zs.w
    # Also test equivalence via potts_to_gumbel
    @test rbm_g_zs.visible.par ≈ potts_to_gumbel(rbm_p_zs).visible.par
    @test rbm_g_zs.hidden.par ≈ potts_to_gumbel(rbm_p_zs).hidden.par
end

@testset "zerosum! PottsGumbel variants" begin
    # visible PottsGumbel, hidden Binary
    N = (3, 2, 3)
    M = (2, 3)
    rbm_g = RBM(PottsGumbel(; θ = randn(N...)), Binary(; θ = randn(M...)), randn(N..., M...))
    v = sample_from_inputs(rbm_g.visible, zeros(N..., 100))
    F_before = free_energy(rbm_g, v)
    rbm_g_zs = zerosum!(rbm_g)
    @test rbm_g_zs.visible isa PottsGumbel

    # hidden PottsGumbel, visible Binary
    N2 = (2, 3)
    M2 = (3, 2, 3)
    rbm_g2 = RBM(Binary(; θ = randn(N2...)), PottsGumbel(; θ = randn(M2...)), randn(N2..., M2...))
    rbm_g2_zs = zerosum!(rbm_g2)
    @test rbm_g2_zs.hidden isa PottsGumbel

    # both PottsGumbel
    N3 = (3, 2, 3)
    M3 = (3, 2, 3)
    rbm_g3 = RBM(PottsGumbel(; θ = randn(N3...)), PottsGumbel(; θ = randn(M3...)), randn(N3..., M3...))
    rbm_g3_zs = zerosum!(rbm_g3)
    @test rbm_g3_zs.visible isa PottsGumbel
    @test rbm_g3_zs.hidden isa PottsGumbel
end

@testset "zerosum! ∂RBM PottsGumbel variants" begin
    # visible PottsGumbel, hidden Binary
    N = (3, 2, 3)
    M = (2, 3)
    rbm_g = RBM(PottsGumbel(; θ = randn(N...)), Binary(; θ = randn(M...)), randn(N..., M...))
    zerosum!(rbm_g)
    v = sample_from_inputs(rbm_g.visible, zeros(N..., 50))
    ∂ = ∂free_energy(rbm_g, v)
    zerosum!(∂, rbm_g)
    @test norm(mean(∂.visible; dims=1)) < 1e-13
    @test norm(mean(∂.w; dims=1)) < 1e-13

    # hidden PottsGumbel, visible Binary
    N2 = (2, 3)
    M2 = (3, 2, 3)
    rbm_g2 = RBM(Binary(; θ = randn(N2...)), PottsGumbel(; θ = randn(M2...)), randn(N2..., M2...))
    zerosum!(rbm_g2)
    v2 = sample_from_inputs(rbm_g2.visible, zeros(N2..., 50))
    ∂2 = ∂free_energy(rbm_g2, v2)
    zerosum!(∂2, rbm_g2)
    @test norm(mean(∂2.hidden; dims=1)) < 1e-13
    @test norm(mean(∂2.w; dims=ndims(rbm_g2.visible) + 1)) < 1e-13

    # both PottsGumbel
    N3 = (3, 2, 3)
    M3 = (3, 2, 3)
    rbm_g3 = RBM(PottsGumbel(; θ = randn(N3...)), PottsGumbel(; θ = randn(M3...)), randn(N3..., M3...))
    zerosum!(rbm_g3)
    v3 = sample_from_inputs(rbm_g3.visible, zeros(N3..., 50))
    ∂3 = ∂free_energy(rbm_g3, v3)
    zerosum!(∂3, rbm_g3)
    @test norm(mean(∂3.visible; dims=1)) < 1e-13
    @test norm(mean(∂3.hidden; dims=1)) < 1e-13
    @test norm(mean(∂3.w; dims=1)) < 1e-13
    @test norm(mean(∂3.w; dims=ndims(rbm_g3.visible) + 1)) < 1e-13
end

@testset "zerosum_weights PottsGumbel variants" begin
    # visible PottsGumbel, hidden Binary
    N = (3, 2, 3)
    M = (2, 3)
    rbm_g = RBM(PottsGumbel(; θ = randn(N...)), Binary(; θ = randn(M...)), randn(N..., M...))
    w = randn(size(rbm_g.w))
    result = zerosum_weights(w, rbm_g)
    result_p = zerosum_weights(w, gumbel_to_potts(rbm_g))
    @test result ≈ result_p

    # hidden PottsGumbel, visible Binary
    N2 = (2, 3)
    M2 = (3, 2, 3)
    rbm_g2 = RBM(Binary(; θ = randn(N2...)), PottsGumbel(; θ = randn(M2...)), randn(N2..., M2...))
    w2 = randn(size(rbm_g2.w))
    result2 = zerosum_weights(w2, rbm_g2)
    result2_p = zerosum_weights(w2, gumbel_to_potts(rbm_g2))
    @test result2 ≈ result2_p

    # both PottsGumbel
    N3 = (3, 2, 3)
    M3 = (3, 2, 3)
    rbm_g3 = RBM(PottsGumbel(; θ = randn(N3...)), PottsGumbel(; θ = randn(M3...)), randn(N3..., M3...))
    w3 = randn(size(rbm_g3.w))
    result3 = zerosum_weights(w3, rbm_g3)
    result3_p = zerosum_weights(w3, gumbel_to_potts(rbm_g3))
    @test result3 ≈ result3_p
end
