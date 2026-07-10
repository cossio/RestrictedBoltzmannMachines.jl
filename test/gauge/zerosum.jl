using Test: @test, @testset
using LinearAlgebra: norm
using Statistics: mean
using RestrictedBoltzmannMachines: zerosum, zerosum!, zerosum_weights, free_energy,
    RBM, Potts, Binary, Spin, Gaussian, ReLU, dReLU, pReLU, xReLU, sample_from_inputs,
    PottsGumbel, potts_to_gumbel, gumbel_to_potts, ∂RBM, ∂free_energy,
    standardize, unstandardize, CenteredRBM, center, uncenter

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

@testset "zerosum StandardizedRBM with nontrivial offsets/scales (visible Potts)" begin
    N = (3, 2)
    M = (2,)
    rbm = RBM(Potts(; θ = randn(N...)), Binary(; θ = randn(M...)), randn(N..., M...))
    srbm = standardize(rbm)
    srbm.offset_v .= randn(N...) / 3
    srbm.scale_v .= 1 .+ rand(N...)
    srbm.offset_h .= randn(M...) / 3
    srbm.scale_h .= 1 .+ rand(M...)

    v = sample_from_inputs(srbm.visible, zeros(N..., 1000))
    F0 = free_energy(srbm, v)

    srbm1 = zerosum(srbm)
    # gauge transformation: free energies shift by an additive constant only
    F1 = free_energy(srbm1, v)
    @test all(F0 - F1 .≈ mean(F0 - F1))
    # offsets and scales are preserved
    @test srbm1.offset_v == srbm.offset_v
    @test srbm1.offset_h == srbm.offset_h
    @test srbm1.scale_v == srbm.scale_v
    @test srbm1.scale_h == srbm.scale_h
    # the equivalent unstandardized RBM is in zerosum gauge
    urbm = unstandardize(srbm1)
    @test norm(mean(urbm.w; dims=1)) < 1e-10
    @test norm(mean(urbm.visible.θ; dims=1)) < 1e-10
    # idempotent
    srbm2 = zerosum(srbm1)
    @test srbm2.w ≈ srbm1.w
    @test srbm2.visible.par ≈ srbm1.visible.par
    @test srbm2.hidden.par ≈ srbm1.hidden.par

    # in-place version agrees with out-of-place version
    srbm! = deepcopy(srbm)
    zerosum!(srbm!)
    @test srbm!.w ≈ srbm1.w
    @test srbm!.visible.par ≈ srbm1.visible.par
    @test srbm!.hidden.par ≈ srbm1.hidden.par
    @test srbm!.offset_v == srbm.offset_v
    @test srbm!.scale_v == srbm.scale_v
end

@testset "zerosum StandardizedRBM with nontrivial offsets/scales (hidden Potts)" begin
    N = (2,)
    M = (3, 2)
    rbm = RBM(Binary(; θ = randn(N...)), Potts(; θ = randn(M...)), randn(N..., M...))
    srbm = standardize(rbm)
    srbm.offset_v .= randn(N...) / 3
    srbm.scale_v .= 1 .+ rand(N...)
    srbm.offset_h .= randn(M...) / 3
    srbm.scale_h .= 1 .+ rand(M...)

    v = sample_from_inputs(srbm.visible, zeros(N..., 1000))
    F0 = free_energy(srbm, v)
    srbm1 = zerosum(srbm)
    F1 = free_energy(srbm1, v)
    @test all(F0 - F1 .≈ mean(F0 - F1))
    urbm = unstandardize(srbm1)
    @test norm(mean(urbm.w; dims=ndims(srbm.visible) + 1)) < 1e-10
    @test norm(mean(urbm.hidden.θ; dims=1)) < 1e-10

    srbm! = deepcopy(srbm)
    zerosum!(srbm!)
    @test free_energy(srbm!, v) ≈ F1
end

@testset "zerosum StandardizedRBM no-op for non-Potts" begin
    srbm = standardize(RBM(Binary(; θ = randn(2)), Binary(; θ = randn(3)), randn(2, 3)))
    srbm.offset_v .= randn(2) / 3
    @test zerosum(srbm) === srbm
    w = copy(srbm.w)
    zerosum!(srbm)
    @test srbm.w == w
end

@testset "zerosum CenteredRBM with nontrivial offsets" begin
    N = (3, 2)
    M = (2,)
    rbm = RBM(Potts(; θ = randn(N...)), Binary(; θ = randn(M...)), randn(N..., M...))
    crbm = CenteredRBM(deepcopy(rbm))
    crbm.offset_v .= randn(N...) / 3
    crbm.offset_h .= randn(M...) / 3

    v = sample_from_inputs(crbm.visible, zeros(N..., 1000))
    F0 = free_energy(crbm, v)

    crbm1 = zerosum(crbm)
    F1 = free_energy(crbm1, v)
    @test all(F0 - F1 .≈ mean(F0 - F1))
    @test crbm1.offset_v == crbm.offset_v
    @test crbm1.offset_h == crbm.offset_h
    # the equivalent uncentered RBM is in zerosum gauge
    urbm = uncenter(crbm1)
    @test norm(mean(urbm.w; dims=1)) < 1e-10
    @test norm(mean(urbm.visible.θ; dims=1)) < 1e-10

    crbm! = deepcopy(crbm)
    zerosum!(crbm!)
    @test crbm!.w ≈ crbm1.w
    @test crbm!.visible.par ≈ crbm1.visible.par
    @test crbm!.hidden.par ≈ crbm1.hidden.par
    @test free_energy(crbm!, v) ≈ F1
end

@testset "zerosum CenteredRBM no-op for non-Potts" begin
    crbm = CenteredRBM(RBM(Binary(; θ = randn(2)), Binary(; θ = randn(3)), randn(2, 3)))
    crbm.offset_v .= randn(2) / 3
    @test zerosum(crbm) === crbm
end

@testset "zerosum StandardizedRBM with nontrivial offsets/scales (both Potts)" begin
    N = (3, 2)
    M = (3, 2)
    rbm = RBM(Potts(; θ = randn(N...)), Potts(; θ = randn(M...)), randn(N..., M...))
    srbm = standardize(rbm)
    srbm.offset_v .= randn(N...) / 3
    srbm.scale_v .= 1 .+ rand(N...)
    srbm.offset_h .= randn(M...) / 3
    srbm.scale_h .= 1 .+ rand(M...)

    v = sample_from_inputs(srbm.visible, zeros(N..., 1000))
    F0 = free_energy(srbm, v)
    srbm1 = zerosum(srbm)
    F1 = free_energy(srbm1, v)
    @test all(F0 - F1 .≈ mean(F0 - F1))
    urbm = unstandardize(srbm1)
    @test norm(mean(urbm.w; dims=1)) < 1e-10
    @test norm(mean(urbm.w; dims=ndims(srbm.visible) + 1)) < 1e-10
    @test norm(mean(urbm.visible.θ; dims=1)) < 1e-10
    @test norm(mean(urbm.hidden.θ; dims=1)) < 1e-10

    # in-place (sequential passes) agrees with out-of-place (joint round trip)
    srbm! = deepcopy(srbm)
    zerosum!(srbm!)
    @test srbm!.w ≈ srbm1.w
    @test srbm!.visible.par ≈ srbm1.visible.par
    @test srbm!.hidden.par ≈ srbm1.hidden.par
end

@testset "zerosum! StandardizedRBM with PottsGumbel visible" begin
    N = (3, 2)
    M = (2,)
    rbm_g = RBM(PottsGumbel(; θ = randn(N...)), Binary(; θ = randn(M...)), randn(N..., M...))
    srbm_g = standardize(rbm_g)
    srbm_g.offset_v .= randn(N...) / 3
    srbm_g.scale_v .= 1 .+ rand(N...)
    srbm_g.offset_h .= randn(M...) / 3
    srbm_g.scale_h .= 1 .+ rand(M...)

    v = sample_from_inputs(srbm_g.visible, zeros(N..., 1000))
    F0 = free_energy(srbm_g, v)
    srbm_g! = deepcopy(srbm_g)
    zerosum!(srbm_g!)
    @test srbm_g!.visible isa PottsGumbel
    F1 = free_energy(srbm_g!, v)
    @test all(F0 - F1 .≈ mean(F0 - F1))
    # agrees with out-of-place
    srbm_g1 = zerosum(srbm_g)
    @test srbm_g!.w ≈ srbm_g1.w
    @test srbm_g!.visible.par ≈ srbm_g1.visible.par
    @test srbm_g!.hidden.par ≈ srbm_g1.hidden.par
end
