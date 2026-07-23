using Random: bitrand
using RestrictedBoltzmannMachines: ∂free_energy
using RestrictedBoltzmannMachines: ∂regularize!
using RestrictedBoltzmannMachines: batchmean
using RestrictedBoltzmannMachines: BinaryRBM
using RestrictedBoltzmannMachines: center
using RestrictedBoltzmannMachines: center!
using RestrictedBoltzmannMachines: center_from_data!
using RestrictedBoltzmannMachines: center_hidden
using RestrictedBoltzmannMachines: center_hidden!
using RestrictedBoltzmannMachines: center_hidden_from_data!
using RestrictedBoltzmannMachines: center_visible
using RestrictedBoltzmannMachines: center_visible!
using RestrictedBoltzmannMachines: center_visible_from_data!
using RestrictedBoltzmannMachines: CenteredBinaryRBM
using RestrictedBoltzmannMachines: CenteredRBM
using RestrictedBoltzmannMachines: ReLU
using RestrictedBoltzmannMachines: dReLU
using RestrictedBoltzmannMachines: Binary
using RestrictedBoltzmannMachines: energy
using RestrictedBoltzmannMachines: free_energy
using RestrictedBoltzmannMachines: initialize!
using RestrictedBoltzmannMachines: inputs_h_from_v
using RestrictedBoltzmannMachines: inputs_v_from_h
using RestrictedBoltzmannMachines: interaction_energy
using RestrictedBoltzmannMachines: mean_h_from_v
using RestrictedBoltzmannMachines: mean_v_from_h
using RestrictedBoltzmannMachines: mirror
using RestrictedBoltzmannMachines: pcd!
using RestrictedBoltzmannMachines: sample_h_from_h
using RestrictedBoltzmannMachines: sample_h_from_v
using RestrictedBoltzmannMachines: sample_v_from_v
using RestrictedBoltzmannMachines: uncenter
using Optimisers: Adam
using Statistics: mean
using LinearAlgebra: norm
using Test: @inferred
using Test: @test
using Test: @testset
using Zygote: gradient

@testset "CenteredBinaryRBM" begin
    rbm = @inferred CenteredBinaryRBM(randn(5), randn(3), randn(5, 3), randn(5), randn(3))
    v = bitrand(size(rbm.visible))
    h = bitrand(size(rbm.hidden))
    E = -rbm.visible.θ' * v - rbm.hidden.θ' * h - (v - rbm.offset_v)' * rbm.w * (h - rbm.offset_h)
    @test @inferred(energy(rbm, v, h)) ≈ E
    @test @inferred(inputs_v_from_h(rbm, h)) ≈ rbm.w * (h - rbm.offset_h)
    @test @inferred(inputs_h_from_v(rbm, v)) ≈ rbm.w' * (v - rbm.offset_v)

    rbm_ = deepcopy(rbm)
    offset_v = randn(size(rbm.visible)...)
    offset_h = randn(size(rbm.hidden)...)
    center!(rbm_, offset_v, offset_h)
    @test rbm_.visible.θ == center(rbm, offset_v, offset_h).visible.θ
    @test rbm_.hidden.θ == center(rbm, offset_v, offset_h).hidden.θ
    @test rbm_.w == center(rbm, offset_v, offset_h).w
    @test rbm_.offset_v == center(rbm, offset_v, offset_h).offset_v == offset_v
end

@testset "center / uncenter" begin
    rbm = BinaryRBM(randn(3), randn(2), randn(3, 2))
    offset_v = randn(3)
    offset_h = randn(2)
    centered_rbm = @inferred center(rbm, offset_v, offset_h)
    @test centered_rbm.offset_v ≈ offset_v
    @test centered_rbm.offset_h ≈ offset_h
    @test @inferred(uncenter(centered_rbm)).visible.θ ≈ rbm.visible.θ
    @test @inferred(uncenter(centered_rbm)).hidden.θ ≈ rbm.hidden.θ
    @test @inferred(uncenter(centered_rbm)).w ≈ rbm.w

    v = bitrand(3, 2)
    h = bitrand(2, 2)
    @test mean_h_from_v(rbm, v) ≈ mean_h_from_v(uncenter(rbm), v)
    @test mean_v_from_h(rbm, h) ≈ mean_v_from_h(uncenter(rbm), h)
end

@testset "center_visible / center_hidden helpers" begin
    rbm = center(BinaryRBM(randn(3), randn(2), randn(3, 2)))
    offset_v = randn(3)
    offset_h = randn(2)

    rbm_visible = @inferred center_visible(rbm, offset_v)
    rbm_hidden = @inferred center_hidden(rbm, offset_h)
    rbm_visible_ref = center(rbm, offset_v, rbm.offset_h)
    rbm_hidden_ref = center(rbm, rbm.offset_v, offset_h)
    @test rbm_visible.visible.par == rbm_visible_ref.visible.par
    @test rbm_visible.hidden.par == rbm_visible_ref.hidden.par
    @test rbm_visible.w == rbm_visible_ref.w
    @test rbm_visible.offset_v == rbm_visible_ref.offset_v
    @test rbm_visible.offset_h == rbm_visible_ref.offset_h
    @test rbm_hidden.visible.par == rbm_hidden_ref.visible.par
    @test rbm_hidden.hidden.par == rbm_hidden_ref.hidden.par
    @test rbm_hidden.w == rbm_hidden_ref.w
    @test rbm_hidden.offset_v == rbm_hidden_ref.offset_v
    @test rbm_hidden.offset_h == rbm_hidden_ref.offset_h

    rbm_visible_mut = deepcopy(rbm)
    @test center_visible!(rbm_visible_mut, offset_v) === rbm_visible_mut
    @test rbm_visible_mut.visible.par == rbm_visible.visible.par
    @test rbm_visible_mut.hidden.par == rbm_visible.hidden.par
    @test rbm_visible_mut.w == rbm_visible.w
    @test rbm_visible_mut.offset_v == rbm_visible.offset_v
    @test rbm_visible_mut.offset_h == rbm_visible.offset_h

    rbm_hidden_mut = deepcopy(rbm)
    @test center_hidden!(rbm_hidden_mut, offset_h) === rbm_hidden_mut
    @test rbm_hidden_mut.visible.par == rbm_hidden.visible.par
    @test rbm_hidden_mut.hidden.par == rbm_hidden.hidden.par
    @test rbm_hidden_mut.w == rbm_hidden.w
    @test rbm_hidden_mut.offset_v == rbm_hidden.offset_v
    @test rbm_hidden_mut.offset_h == rbm_hidden.offset_h
end

@testset "rbm energy invariance" begin
    centered_rbm = CenteredBinaryRBM(randn(3), randn(2), randn(3, 2), randn(3), randn(2))
    rbm = uncenter(centered_rbm)
    ΔE = interaction_energy(rbm, centered_rbm.offset_v, centered_rbm.offset_h)::Number
    v = bitrand(size(rbm.visible)..., 100)
    h = bitrand(size(rbm.hidden)..., 100)
    @test @inferred(energy(centered_rbm, v, h)) ≈ energy(rbm, v, h) .+ ΔE
    @test @inferred(free_energy(centered_rbm, v)) ≈ free_energy(rbm, v) .+ ΔE
end

@testset "free energy" begin
    rbm = CenteredBinaryRBM(randn(3), randn(2), randn(3, 2), randn(3), randn(2))
    v = bitrand(size(rbm.visible)...)
    F = -log(sum(exp(-energy(rbm, v, h)) for h in [[0, 0], [0, 1], [1, 0], [1, 1]]))
    @test @inferred(free_energy(rbm, v)) ≈ F
end

@testset "∂free energy" begin
    rbm = CenteredBinaryRBM(randn(3), randn(2), randn(3, 2), randn(3), randn(2))
    v = bitrand(size(rbm.visible)...)
    gs = gradient(rbm) do rbm
        mean(free_energy(rbm, v))
    end
    ∂ = @inferred ∂free_energy(rbm, v)
    @test ∂.visible ≈ only(gs).visible.par
    @test ∂.hidden ≈ only(gs).hidden.par
    @test ∂.w ≈ only(gs).w
end

using RestrictedBoltzmannMachines: free_energy_h, free_energy_v, ∂free_energy_h, ∂free_energy_v

@testset "free_energy_h of CenteredRBM" begin
    centered_rbm = CenteredBinaryRBM(randn(3), randn(2), randn(3, 2), randn(3), randn(2))
    rbm = uncenter(centered_rbm)
    ΔE = interaction_energy(rbm, centered_rbm.offset_v, centered_rbm.offset_h)::Number
    h = bitrand(size(centered_rbm.hidden)..., 100)
    # consistent with the equivalent uncentered RBM, up to the constant energy shift
    @test @inferred(free_energy_h(centered_rbm, h)) ≈ free_energy_h(rbm, h) .+ ΔE
    # consistent with marginalizing the mirrored model
    @test free_energy_h(centered_rbm, h) ≈ free_energy(mirror(centered_rbm), h)
    # exact marginalization over visible states
    h1 = bitrand(size(centered_rbm.hidden)...)
    vs = ([a, b, c] for a in 0:1, b in 0:1, c in 0:1)
    F = -log(sum(exp(-energy(centered_rbm, v, h1)) for v in vs))
    @test free_energy_h(centered_rbm, h1) ≈ F
end

@testset "∂free_energy_v / ∂free_energy_h of CenteredRBM" begin
    rbm = CenteredBinaryRBM(randn(3), randn(2), randn(3, 2), randn(3), randn(2))
    v = bitrand(size(rbm.visible)..., 10)
    h = bitrand(size(rbm.hidden)..., 10)

    @test free_energy_v(rbm, v) == free_energy(rbm, v)
    @test ∂free_energy_v(rbm, v) == ∂free_energy(rbm, v)

    gs_h = gradient(rbm) do rbm
        mean(free_energy_h(rbm, h))
    end
    ∂h = ∂free_energy_h(rbm, h)
    @test ∂h.visible ≈ only(gs_h).visible.par
    @test ∂h.hidden ≈ only(gs_h).hidden.par
    @test ∂h.w ≈ only(gs_h).w
end

@testset "mirror" begin
    rbm = CenteredBinaryRBM(
        randn(5, 2), randn(7, 4, 3), randn(5, 2, 7, 4, 3), randn(5, 2), randn(7, 4, 3)
    )
    rbm_mirror = @inferred mirror(rbm)
    @test rbm_mirror.visible == rbm.hidden
    @test rbm_mirror.hidden == rbm.visible
    @test rbm_mirror.offset_v == rbm.offset_h
    @test rbm_mirror.offset_h == rbm.offset_v
    v = rand(Bool, size(rbm.visible)..., 13)
    h = rand(Bool, size(rbm.hidden)..., 13)
    @test energy(rbm_mirror, h, v) ≈ energy(rbm, v, h)
end

@testset "centered pcd" begin
    rbm = center(BinaryRBM((28, 28), 100))
    train_x = bitrand(28, 28, 1024)

    initialize!(rbm, train_x) # fit independent site statistics and center
    @test rbm.offset_v ≈ dropdims(mean(train_x; dims = 3); dims = 3)
    train_h = mean_h_from_v(rbm, train_x)
    @test rbm.offset_h ≈ vec(mean(train_h; dims = 2))
    @test norm(mean(inputs_h_from_v(rbm, train_x); dims = 2)) < 1.0e-6

    pcd!(rbm, train_x; batchsize = 1024, iters = 100)
    @test rbm.offset_v ≈ dropdims(mean(train_x; dims = 3); dims = 3)
    train_h = mean_h_from_v(rbm, train_x)
    @test rbm.offset_h ≈ vec(mean(train_h; dims = 2)) rtol = 0.1
    # not exact because offset is updated after having updated the parameters!
end

@testset "centered pcd training" begin
    data = falses(2, 1000)
    data[1, 1:2:end] .= true
    data[2, 1:2:end] .= true

    rbm = center(BinaryRBM(2, 5))
    initialize!(rbm, data)
    pcd!(rbm, data; iters = 10000, batchsize = 64, steps = 10, optim = Adam(5.0e-4))

    v_sample = sample_v_from_v(rbm, bitrand(2, 10000); steps = 50)

    @test 0.4 < mean(v_sample[1, :]) < 0.6
    @test 0.4 < mean(v_sample[2, :]) < 0.6
    @test 0.4 < mean(v_sample[1, :] .* v_sample[2, :]) < 0.6
end

@testset "center_from_data! helpers" begin
    rbm = center(BinaryRBM(randn(3), randn(2), randn(3, 2)))
    data = bitrand(3, 7)
    wts = rand(7)

    rbm_visible = deepcopy(rbm)
    @test center_visible_from_data!(rbm_visible, data; wts) === rbm_visible
    expected_offset_v = batchmean(rbm.visible, data; wts)
    @test rbm_visible.offset_v ≈ expected_offset_v

    expected_hidden = center_visible(rbm, expected_offset_v)
    expected_offset_h = batchmean(expected_hidden.hidden, mean_h_from_v(expected_hidden, data); wts)

    rbm_hidden = deepcopy(expected_hidden)
    @test center_hidden_from_data!(rbm_hidden, data; wts) === rbm_hidden
    @test rbm_hidden.offset_h ≈ expected_offset_h

    rbm_data = deepcopy(rbm)
    @test center_from_data!(rbm_data, data; wts) === rbm_data
    @test rbm_data.visible.par == rbm_hidden.visible.par
    @test rbm_data.hidden.par == rbm_hidden.hidden.par
    @test rbm_data.w == rbm_hidden.w
    @test rbm_data.offset_v == rbm_hidden.offset_v
    @test rbm_data.offset_h == rbm_hidden.offset_h
end

@testset "centered pcd uses weighted initial centering" begin
    rbm = center(BinaryRBM(2, 3))
    data = falses(2, 4)
    data[:, 3:4] .= true
    wts = [100.0, 100.0, 1.0, 1.0]

    weighted_offset_v = batchmean(rbm.visible, data; wts)
    weighted_hidden = center_visible(rbm, weighted_offset_v)
    weighted_offset_h = batchmean(weighted_hidden.hidden, mean_h_from_v(weighted_hidden, data); wts)

    initial_offset_v = Ref{Any}()
    initial_offset_h = Ref{Any}()
    seen = Ref(false)

    pcd!(
        rbm,
        data;
        wts,
        batchsize = 2,
        iters = 1,
        steps = 0,
        hidden_offset_damping = 0,
        callback = (; rbm, iter, kwargs...) -> begin
            if iter == 1 && !seen[]
                initial_offset_v[] = copy(rbm.offset_v)
                initial_offset_h[] = copy(rbm.offset_h)
                seen[] = true
            end
            nothing
        end,
    )

    @test seen[]
    @test initial_offset_v[] ≈ weighted_offset_v
    @test initial_offset_h[] ≈ weighted_offset_h
end

@testset "sample_h_from_h centered RBM" begin
    rbm = center(BinaryRBM(randn(3), randn(2), zeros(3, 2)))
    h = bitrand(2, 10^5)
    v = falses(3, 10^5)
    sample = @inferred sample_h_from_h(rbm, h)
    @test size(sample) == size(h)
    @test batchmean(rbm.hidden, sample) ≈ batchmean(rbm.hidden, mean_h_from_v(rbm, v)) rtol = 0.1
end

@testset "∂regularize! centered RBM" begin
    rbm = CenteredBinaryRBM(randn(3), randn(2), randn(3, 2), randn(3), randn(2))
    v = bitrand(3, 100)
    vdims = ntuple(identity, ndims(rbm.visible))
    N = length(rbm.visible)

    l2_fields = rand()
    l1_weights = rand()
    l2_weights = rand()
    l2l1_weights = rand()

    urbm = uncenter(rbm)
    ∂u = ∂free_energy(urbm, v)

    gs = gradient(rbm) do rbm
        F = mean(free_energy(rbm, v))
        urbm = uncenter(rbm)
        L2_fields = sum(abs2, urbm.visible.θ)
        L1_weights = sum(abs, urbm.w)
        L2_weights = sum(abs2, urbm.w)
        L2L1_weights = sum(abs2, sum(abs, urbm.w; dims = vdims))
        return (
            F + l2_fields / 2 * L2_fields +
                l1_weights * L1_weights +
                l2_weights / 2 * L2_weights +
                l2l1_weights / (2N) * L2L1_weights
        )
    end

    ∂ = ∂free_energy(rbm, v)
    ∂regularize!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights)

    @test only(gs).visible.par ≈ ∂.visible
    @test only(gs).hidden.par ≈ ∂.hidden
    @test only(gs).w ≈ ∂.w
end

@testset "∂regularize! centered RBM" begin
    rbm = CenteredRBM(
        ReLU(; θ = randn(3), γ = rand(3)), Binary(; θ = randn(2)), randn(3, 2),
        randn(3), randn(2)
    )
    v = rand(3, 100)
    vdims = ntuple(identity, ndims(rbm.visible))
    N = length(rbm.visible)

    l2_fields = rand()
    l1_weights = rand()
    l2_weights = rand()
    l2l1_weights = rand()

    gs = gradient(rbm) do rbm
        F = mean(free_energy(rbm, v))
        urbm = uncenter(rbm)
        L2_fields = sum(abs2, urbm.visible.θ)
        L1_weights = sum(abs, urbm.w)
        L2_weights = sum(abs2, urbm.w)
        L2L1_weights = sum(abs2, sum(abs, urbm.w; dims = vdims))
        return (
            F + l2_fields / 2 * L2_fields +
                l1_weights * L1_weights +
                l2_weights / 2 * L2_weights +
                l2l1_weights / (2N) * L2L1_weights
        )
    end

    ∂ = ∂free_energy(rbm, v)
    ∂regularize!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights)

    @test only(gs).visible.par ≈ ∂.visible
    @test only(gs).hidden.par ≈ ∂.hidden
    @test only(gs).w ≈ ∂.w
end

@testset "∂regularize! centered RBM" begin
    rbm = CenteredRBM(
        dReLU(; θp = randn(3), θn = randn(3), γp = rand(3), γn = rand(3)), Binary(; θ = randn(2)), randn(3, 2),
        randn(3), randn(2),
    )
    v = randn(3, 100)
    vdims = ntuple(identity, ndims(rbm.visible))
    N = length(rbm.visible)

    l2_fields = rand()
    l1_weights = rand()
    l2_weights = rand()
    l2l1_weights = rand()

    gs = gradient(rbm) do rbm
        F = mean(free_energy(rbm, v))
        urbm = uncenter(rbm)
        L2_fields = sum(abs2, urbm.visible.θp) + sum(abs2, urbm.visible.θn)
        L1_weights = sum(abs, urbm.w)
        L2_weights = sum(abs2, urbm.w)
        L2L1_weights = sum(abs2, sum(abs, urbm.w; dims = vdims))
        return (
            F + l2_fields / 2 * L2_fields +
                l1_weights * L1_weights +
                l2_weights / 2 * L2_weights +
                l2l1_weights / (2N) * L2L1_weights
        )
    end

    ∂ = ∂free_energy(rbm, v)
    ∂regularize!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights)

    @test only(gs).visible.par ≈ ∂.visible
    @test only(gs).hidden.par ≈ ∂.hidden
    @test only(gs).w ≈ ∂.w
end

using RestrictedBoltzmannMachines: RBM, regularization_penalty

@testset "regularization_penalty of CenteredRBM" begin
    rbm = CenteredBinaryRBM(randn(3), randn(2), randn(3, 2), randn(3), randn(2))
    l2_fields, l1_weights, l2_weights, l2l1_weights = rand(4)
    @test regularization_penalty(rbm; l2_fields, l1_weights, l2_weights, l2l1_weights) ≈
        regularization_penalty(uncenter(rbm); l2_fields, l1_weights, l2_weights, l2l1_weights)
    @test regularization_penalty(rbm; regularize_unstandardized = false, l2_fields, l1_weights, l2_weights, l2l1_weights) ≈
        regularization_penalty(RBM(rbm); l2_fields, l1_weights, l2_weights, l2l1_weights)
end

using RestrictedBoltzmannMachines: log_pseudolikelihood

@testset "CenteredBinaryRBM with zero offsets" begin
    a, b, w = randn(3), randn(2), randn(3, 2)
    rbm = @inferred CenteredBinaryRBM(a, b, w)
    @test iszero(rbm.offset_v)
    @test iszero(rbm.offset_h)
    v = bitrand(3, 5)
    h = bitrand(2, 5)
    @test energy(rbm, v, h) ≈ energy(BinaryRBM(a, b, w), v, h)
end

@testset "conditional means of CenteredRBM" begin
    centered_rbm = CenteredBinaryRBM(randn(3), randn(2), randn(3, 2), randn(3), randn(2))
    rbm = uncenter(centered_rbm)
    v = bitrand(3, 5)
    h = bitrand(2, 5)
    @test @inferred(mean_h_from_v(centered_rbm, v)) ≈ mean_h_from_v(rbm, v)
    @test @inferred(mean_v_from_h(centered_rbm, h)) ≈ mean_v_from_h(rbm, h)
end

@testset "center_visible / center_hidden from a plain RBM" begin
    rbm = BinaryRBM(randn(3), randn(2), randn(3, 2))
    offset_v = randn(3)
    offset_h = randn(2)
    crbm_v = @inferred center_visible(rbm, offset_v)
    crbm_h = @inferred center_hidden(rbm, offset_h)
    @test crbm_v.offset_v == offset_v
    @test iszero(crbm_v.offset_h)
    @test crbm_h.offset_h == offset_h
    @test iszero(crbm_h.offset_v)
    # centering is a gauge transformation: free energies shift by a constant
    v = bitrand(3, 7)
    F0 = free_energy(rbm, v)
    for crbm in (crbm_v, crbm_h)
        F = free_energy(crbm, v)
        @test F ≈ F0 .+ mean(F - F0)
    end
end

@testset "center! without arguments resets offsets" begin
    rbm = CenteredBinaryRBM(randn(3), randn(2), randn(3, 2), randn(3), randn(2))
    v = bitrand(3, 7)
    F0 = free_energy(rbm, v)
    @test center!(rbm) === rbm
    @test iszero(rbm.offset_v)
    @test iszero(rbm.offset_h)
    F1 = free_energy(rbm, v)
    @test F1 ≈ F0 .+ mean(F1 - F0)
end

@testset "log_pseudolikelihood of CenteredRBM" begin
    # With a single visible site the stochastic estimator is deterministic,
    # and centering must not change the pseudolikelihood.
    rbm = CenteredBinaryRBM(randn(1), randn(2), randn(1, 2), randn(1), randn(2))
    v = bitrand(1, 7)
    @test log_pseudolikelihood(rbm, v) ≈ log_pseudolikelihood(uncenter(rbm), v)
    @test log_pseudolikelihood(rbm, v; exact = true) ≈
        log_pseudolikelihood(uncenter(rbm), v; exact = true)
end

using RestrictedBoltzmannMachines: RBM, rescale_hidden!, rescale_weights!, weight_norms

@testset "rescale_hidden! and rescale_weights! of CenteredRBM" begin
    #= rescale_hidden! reparameterizes the hidden units so that activations are divided
    by λ. Since the interaction involves h - offset_h, the hidden offsets must be divided
    by λ as well; free energies then shift by the constant log-Jacobian sum(log, λ) and
    the modeled distribution is unchanged. =#
    rbm = CenteredRBM(
        Binary(; θ = randn(3)), ReLU(; θ = randn(2), γ = 0.5 .+ rand(2)), randn(3, 2),
        randn(3), randn(2),
    )
    rbm_copy = deepcopy(rbm)
    v = bitrand(size(rbm.visible)..., 100)
    λ = 0.5 .+ rand(size(rbm.hidden)...)

    @test @inferred rescale_hidden!(rbm, λ)
    @test rbm.offset_h ≈ rbm_copy.offset_h ./ λ
    # free energies shift by the constant log-Jacobian, sum(log, λ)
    @test free_energy(rbm, v) ≈ free_energy(rbm_copy, v) .+ sum(log, λ)

    rbm = deepcopy(rbm_copy)
    ω = @inferred weight_norms(rbm)
    @test ω ≈ weight_norms(RBM(rbm))
    @test @inferred rescale_weights!(rbm)
    @test weight_norms(rbm) ≈ ones(size(rbm.hidden))
    @test free_energy(rbm, v) ≈ free_energy(rbm_copy, v) .- sum(log, ω)

    # discrete hidden units have no scale parameter: no-op, returns false
    rbm = CenteredBinaryRBM(randn(3), randn(2), randn(3, 2), randn(3), randn(2))
    rbm_copy = deepcopy(rbm)
    @test !rescale_hidden!(rbm, 0.5 .+ rand(size(rbm.hidden)...))
    @test !rescale_weights!(rbm)
    @test rbm.visible.par == rbm_copy.visible.par
    @test rbm.hidden.par == rbm_copy.hidden.par
    @test rbm.w == rbm_copy.w
    @test rbm.offset_v == rbm_copy.offset_v
    @test rbm.offset_h == rbm_copy.offset_h
end

@testset "rescale_weights! of CenteredRBM preserves zero-norm hidden units" begin
    rbm = CenteredRBM(
        Binary(; θ = randn(2)),
        ReLU(; θ = [0.25, -0.5], γ = [1.5, 2.0]),
        [0.0 3.0; 0.0 4.0],
        randn(2),
        [0.75, -0.25],
    )
    rbm0 = deepcopy(rbm)
    v = Bool[0 0 1 1; 0 1 0 1]
    F0 = free_energy(rbm, v)
    ω = weight_norms(rbm)
    @test ω ≈ [0, 5]

    @test @inferred rescale_weights!(rbm)
    @test weight_norms(rbm) ≈ [0, 1]
    @test rbm.w[:, 1] == rbm0.w[:, 1]
    @test rbm.hidden.par[:, 1] == rbm0.hidden.par[:, 1]
    @test rbm.offset_h[1] == rbm0.offset_h[1]
    @test all(isfinite, rbm.w)
    @test all(isfinite, rbm.hidden.par)
    @test all(isfinite, rbm.offset_h)
    @test free_energy(rbm, v) ≈ F0 .- log(ω[2])
end

using LogExpFunctions: softmax
using RestrictedBoltzmannMachines: Gaussian, pReLU, xReLU, collect_states, var_from_inputs

#= rescale_hidden!(::CenteredRBM, λ) must be a gauge transformation: it reparameterizes
the hidden units as h -> h / λ without changing the modeled distribution p(v). The
offsets are drawn nonzero because offset_h enters the interaction as h - offset_h and
must be rescaled together with the activations for p(v) to be preserved. =#
@testset "rescale_hidden! of CenteredRBM is a gauge transformation ($name hidden)" for (name, hidden) in [
        ("Gaussian", Gaussian(; θ = randn(2), γ = 0.5 .+ rand(2))),
        ("ReLU", ReLU(; θ = randn(2), γ = 0.5 .+ rand(2))),
        ("dReLU", dReLU(; θp = randn(2), θn = randn(2), γp = 0.5 .+ rand(2), γn = 0.5 .+ rand(2))),
        ("pReLU", pReLU(; θ = randn(2), γ = 0.5 .+ rand(2), Δ = randn(2), η = rand(2) .- 0.5)),
        ("xReLU", xReLU(; θ = randn(2), γ = 0.5 .+ rand(2), Δ = randn(2), ξ = randn(2))),
    ]
    rbm0 = CenteredRBM(Binary(; θ = randn(3)), hidden, randn(3, 2), randn(3), randn(2))
    rbm1 = deepcopy(rbm0)
    λ = 0.5 .+ rand(2)
    @test rescale_hidden!(rbm1, λ)

    # the joint energy is preserved under the reparameterization h -> h / λ;
    # draw h from the model so it lies in the hidden layer's support (e.g. h ≥ 0
    # for ReLU) and every sample exercises the finite-energy identity
    v = bitrand(3, 10)
    h = sample_h_from_v(rbm0, v)
    @test energy(rbm1, v, h ./ λ) ≈ energy(rbm0, v, h)

    # exact enumeration of all visible states: free energies shift by the constant
    # log-Jacobian of h -> h / λ, so the visible distribution p(v) is exactly unchanged
    states = collect_states(rbm0.visible)
    F0 = free_energy(rbm0, states)
    F1 = free_energy(rbm1, states)
    @test F1 ≈ F0 .+ sum(log, λ)
    @test softmax(-F1) ≈ softmax(-F0)

    # conditional hidden statistics rescale as h -> h / λ ...
    @test mean_h_from_v(rbm1, v) ≈ mean_h_from_v(rbm0, v) ./ λ
    @test var_from_inputs(rbm1.hidden, inputs_h_from_v(rbm1, v)) ≈
        var_from_inputs(rbm0.hidden, inputs_h_from_v(rbm0, v)) ./ λ .^ 2
    # ... and the conditional visible distribution is preserved
    @test mean_v_from_h(rbm1, h ./ λ) ≈ mean_v_from_h(rbm0, h)

    # rescale_weights! is the λ = 1 ./ weight_norms special case, and must
    # likewise leave p(v) exactly unchanged while normalizing the weights
    rbm2 = deepcopy(rbm0)
    @test rescale_weights!(rbm2)
    @test weight_norms(rbm2) ≈ ones(size(rbm2.hidden))
    F2 = free_energy(rbm2, states)
    @test F2 ≈ F0 .+ mean(F2 - F0) # constant shift
    @test softmax(-F2) ≈ softmax(-F0)
end
