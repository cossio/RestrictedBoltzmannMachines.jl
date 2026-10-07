using RestrictedBoltzmannMachines: ∂RBM, ∂unstandardize, CompositeRegularizer, StandardizedParametersRegularizer, L2FieldsRegularizer,
    L1WeightsRegularizer, L2WeightsRegularizer, L2L1WeightsRegularizer
using LinearAlgebra: norm
using LogExpFunctions: logsumexp
using Random: bitrand, seed!
using RestrictedBoltzmannMachines: ∂free_energy, ∂free_energy_h, ∂free_energy_v, ∂regularize!
using RestrictedBoltzmannMachines: Binary, Spin, dReLU, Potts, ReLU, nsReLU
using RestrictedBoltzmannMachines: RBM, StandardizedRBM
using RestrictedBoltzmannMachines: BinaryRBM, BinaryStandardizedRBM, SpinStandardizedRBM
using RestrictedBoltzmannMachines: weight_norms, delta_energy
using RestrictedBoltzmannMachines: rescale_weights!, rescale_hidden_activations!
using RestrictedBoltzmannMachines: energy, free_energy, free_energy_h, free_energy_v
using RestrictedBoltzmannMachines: generate_sequences
using RestrictedBoltzmannMachines: log_partition
using RestrictedBoltzmannMachines: mean_h_from_v, mean_v_from_h, var_h_from_v, var_v_from_h
using RestrictedBoltzmannMachines: mirror
using RestrictedBoltzmannMachines: pcd!, regularization_penalty
using RestrictedBoltzmannMachines: sample_h_from_h, sample_v_from_v
using RestrictedBoltzmannMachines: standardize, unstandardize, standardize!, unstandardized_weights
using RestrictedBoltzmannMachines: standardize_hidden, standardize_visible
using RestrictedBoltzmannMachines: initialize!
using StatsBase: proportionmap
using Statistics: mean, std, var
using FillArrays: Trues
using Test: @inferred, @test, @testset
using Zygote: gradient

@testset "standardize" begin
    rbm = BinaryRBM(randn(3), randn(2), randn(3, 2))
    offset_v = randn(3)
    offset_h = randn(2)
    scale_v = rand(3)
    scale_h = rand(2)

    std_rbm = @inferred standardize(rbm, offset_v, offset_h, scale_v, scale_h)
    @test std_rbm.offset_v ≈ offset_v
    @test std_rbm.offset_h ≈ offset_h
    @test std_rbm.scale_v ≈ scale_v
    @test std_rbm.scale_h ≈ scale_h

    @test std_rbm.w ./ (reshape(scale_v, 3, 1) .* reshape(scale_h, 1, 2)) ≈ rbm.w

    v = bitrand(3, 10)
    h = bitrand(2, 10)
    @test energy(std_rbm, v, h) .- delta_energy(std_rbm) ≈ energy(rbm, v, h)
    @test free_energy(std_rbm, v) .- delta_energy(std_rbm) ≈ free_energy(rbm, v)

    @test iszero(standardize(rbm).offset_v)
    @test iszero(standardize(rbm).offset_h)
    @test all(standardize(rbm).scale_v .== 1)
    @test all(standardize(rbm).scale_h .== 1)
    @inferred standardize(rbm)
    @test iszero(standardize(std_rbm).offset_v)
    @test iszero(standardize(std_rbm).offset_h)
    @test all(standardize(std_rbm).scale_v .== 1)
    @test all(standardize(std_rbm).scale_h .== 1)
    @inferred standardize(std_rbm)
    @test energy(rbm, v, h) ≈ energy(standardize(rbm), v, h) ≈ energy(standardize(std_rbm), v, h)
end

@testset "unstandardize" begin
    std_rbm = @inferred BinaryStandardizedRBM(randn(3), randn(2), randn(3, 2), randn(3), randn(2), rand(3), rand(2))
    rbm = @inferred unstandardize(std_rbm)
    @test rbm isa RBM
    @test unstandardize(rbm) == rbm
    v = bitrand(3, 10)
    h = bitrand(2, 10)
    @test energy(std_rbm, v, h) .- delta_energy(std_rbm) ≈ energy(rbm, v, h)
end

@testset "delta_energy" begin
    rbm = BinaryRBM(randn(3), randn(2), randn(3, 2))
    std_rbm = @inferred standardize(rbm, randn(3), randn(2), rand(3), rand(2))
    @test iszero(@inferred delta_energy(rbm))
    @test iszero(delta_energy(standardize(std_rbm)))
    @test delta_energy(rbm) isa Real
    @test delta_energy(std_rbm) isa Real
    # the energies of the two parametrizations differ by the constant, and so do their
    # log-partition functions, with the opposite sign
    v = bitrand(3, 10)
    for model in (std_rbm, standardize(rbm, randn(3), randn(2))) # standardized, centered
        @test free_energy(model, v) ≈ free_energy(unstandardize(model), v) .+ delta_energy(model)
        @test log_partition(unstandardize(model)) ≈ log_partition(model) + delta_energy(model)
    end
end

@testset "standardize!" begin
    rbm = @inferred BinaryStandardizedRBM(
        randn(3), randn(2), randn(3, 2),
        randn(3), randn(2), rand(3), rand(2)
    )
    offset_v = randn(3)
    offset_h = randn(2)
    scale_v = rand(3)
    scale_h = rand(2)

    v = bitrand(3, 10)
    h = bitrand(2, 10)
    E = energy(rbm, v, h) .- delta_energy(rbm)
    F = free_energy(rbm, v) .- delta_energy(rbm)

    standardize!(rbm, offset_v, offset_h, scale_v, scale_h)
    @test rbm.offset_v ≈ offset_v
    @test rbm.offset_h ≈ offset_h
    @test rbm.scale_v ≈ scale_v
    @test rbm.scale_h ≈ scale_h

    @test energy(rbm, v, h) .- delta_energy(rbm) ≈ E
    @test free_energy(rbm, v) .- delta_energy(rbm) ≈ F
end

@testset "initialize! StandardizedRBM without data" begin
    rbm = BinaryStandardizedRBM(randn(3), randn(2), randn(3, 2), randn(3), randn(2), rand(3), rand(2))
    @test initialize!(rbm) === rbm
    @test iszero(rbm.offset_v)
    @test iszero(rbm.offset_h)
    @test all(isone, rbm.scale_v)
    @test all(isone, rbm.scale_h)
    @test iszero(rbm.visible.θ)
    @test iszero(rbm.hidden.θ)
    @test !iszero(rbm.w)
end

@testset "initialize! StandardizedRBM with data" begin
    data = bitrand(3, 20)
    # the result is the standardized form of the initialized plain RBM, independent of the
    # previous offsets and scales
    rbm = BinaryStandardizedRBM(randn(3), randn(2), randn(3, 2), randn(3), randn(2), rand(3), rand(2))
    plain = BinaryRBM(randn(3), randn(2), randn(3, 2))
    seed!(1)
    @test initialize!(rbm, data) === rbm
    seed!(1)
    initialize!(plain, data)
    @test unstandardize(rbm).visible.θ ≈ plain.visible.θ
    @test unstandardize(rbm).hidden.θ ≈ plain.hidden.θ atol = 1.0e-12
    @test unstandardize(rbm).w ≈ plain.w
    @test rbm.offset_v ≈ vec(mean(data; dims = 2))
    @test rbm.scale_v ≈ vec(std(data; dims = 2, corrected = false))
    @test rbm.offset_h ≈ vec(mean(mean_h_from_v(rbm, data); dims = 2))
    ν = vec(mean(var_h_from_v(rbm, data); dims = 2) + var(mean_h_from_v(rbm, data); dims = 2, corrected = false))
    @test rbm.scale_h ≈ sqrt.(ν)
end

@testset "∂free energy ($name)" for (name, rbm, h) in (
        ("Binary", BinaryStandardizedRBM(randn(3), randn(2), randn(3, 2), randn(3), randn(2), rand(3), rand(2)), bitrand(2, 10)),
        (
            "nsReLU",
            standardize(
                RBM(Binary(; θ = randn(3)), nsReLU(; θ = randn(2), ξ = randn(2), Δ = randn(2)), randn(3, 2)),
                randn(3), randn(2), 0.1 .+ rand(3), 0.1 .+ rand(2)
            ),
            randn(2, 10),
        ),
    )
    v = bitrand(size(rbm.visible)..., 10)

    @test free_energy_v(rbm, v) == free_energy(rbm, v)
    @test ∂free_energy_v(rbm, v) == ∂free_energy(rbm, v)

    gs_v = gradient(rbm) do rbm
        mean(free_energy_v(rbm, v))
    end
    ∂v = ∂free_energy_v(rbm, v)
    @test ∂v.visible ≈ only(gs_v).visible.par
    @test ∂v.hidden ≈ only(gs_v).hidden.par
    @test ∂v.w ≈ only(gs_v).w

    gs_h = gradient(rbm) do rbm
        mean(free_energy_h(rbm, h))
    end
    ∂h = ∂free_energy_h(rbm, h)
    @test ∂h.visible ≈ only(gs_h).visible.par
    @test ∂h.hidden ≈ only(gs_h).hidden.par
    @test ∂h.w ≈ only(gs_h).w
end

@testset "standardized constructor" begin
    rbm = SpinStandardizedRBM(randn(10), randn(7), randn(10, 7))
    @test rbm.visible isa Spin
    @test rbm.hidden isa Spin
end

@testset "rescale_hidden_activations!" begin
    rbm = standardize(
        RBM(Binary(; θ = randn(3)), ReLU(; θ = randn(2), γ = 0.1 .+ rand(2)), randn(3, 2)),
        randn(3), randn(2), 0.1 .+ rand(3), 0.1 .+ rand(2)
    )

    v = bitrand(3, 1000)
    F = free_energy(rbm, v)
    h_ave = mean_h_from_v(rbm, v)
    h_var = var_h_from_v(rbm, v)
    v_ave = mean_v_from_h(rbm, h_ave)
    v_var = var_v_from_h(rbm, h_ave)

    λ = copy(rbm.scale_h)
    rescale_hidden_activations!(rbm)

    @test mean_h_from_v(rbm, v) ≈ h_ave ./ λ
    @test var_h_from_v(rbm, v) ≈ h_var ./ λ .^ 2
    @test mean_v_from_h(rbm, h_ave ./ λ) ≈ v_ave
    @test var_v_from_h(rbm, h_ave ./ λ) ≈ v_var
    @test all(rbm.scale_h .≈ 1)
    @test free_energy(rbm, v) ≈ F .+ sum(log, λ)
end

@testset "standardized pcd" begin
    rbm = standardize(RBM(Spin(; θ = zeros(10)), Spin(; θ = zeros(7)), randn(10, 7) / √10))
    @test iszero(rbm.visible.θ) && iszero(rbm.hidden.θ)

    data = ones(10, 4)
    data[1:3, 2] .= -1
    data[:, 3:4] .= -data[:, 1:2] # ensure data has zero mean
    @test iszero(mean(data; dims = 2))

    state, ps = pcd!(
        rbm, data;
        ps = (; w = rbm.w), # train only weights
        steps = 10, batchsize = 4, iters = 1000,
        ϵv = 1.0f-1, ϵh = 0.0f0, damping = 1.0f-1
    )

    # The fields are not exactly zero because centering introduces minor numerical fluctuations.
    @test norm(rbm.visible.θ) < 1.0e-13
    @test iszero(rbm.hidden.θ)
end

@testset "standardized pcd with zero-variance visible features" begin
    binary_data = Bool[
        0 0 0 0
        0 1 0 1
    ]
    potts_data = reshape(
        Bool[
            1 0 1 0
            0 1 0 1
            0 0 0 0
        ], 3, 1, 4
    )
    # uniform weights of any scale must reproduce the unweighted statistics
    # exactly, since weights are never rescaled and always cancel in the
    # weighted means and variances
    for (
            data_binary, data_potts, wts,
            binary_offset, binary_scale, potts_offset, potts_scale,
        ) in (
            (
                binary_data,
                potts_data,
                Trues(4),
                [0.0, 0.5],
                [1.0, 0.5],
                reshape([0.5, 0.5, 0.0], 3, 1),
                reshape([0.5, 0.5, 1.0], 3, 1),
            ),
            (
                binary_data,
                potts_data,
                fill(2.0, 4),
                [0.0, 0.5],
                [1.0, 0.5],
                reshape([0.5, 0.5, 0.0], 3, 1),
                reshape([0.5, 0.5, 1.0], 3, 1),
            ),
        )
        binary_rbm = standardize(BinaryRBM(zeros(2), zeros(1), fill(0.1, 2, 1)))
        pcd!(
            binary_rbm, data_binary;
            wts, iters = 1, batchsize = 4, steps = 1,
        )
        @test binary_rbm.offset_v ≈ binary_offset
        @test binary_rbm.scale_v ≈ binary_scale

        potts_rbm = standardize(
            RBM(Potts((3, 1)), Binary((1,)), fill(0.1, 3, 1, 1))
        )
        pcd!(
            potts_rbm, data_potts;
            wts, iters = 1, batchsize = 4, steps = 1,
        )
        @test potts_rbm.offset_v ≈ potts_offset
        @test potts_rbm.scale_v ≈ potts_scale

        for (rbm, data) in ((binary_rbm, data_binary), (potts_rbm, data_potts))
            @test all(rbm.scale_v .> 0)
            for par in (
                    rbm.visible.par, rbm.hidden.par, rbm.w,
                    rbm.offset_v, rbm.offset_h, rbm.scale_v, rbm.scale_h,
                )
                @test all(isfinite, par)
            end
            @test all(isfinite, free_energy(rbm, data))
        end
    end

    rbm = standardize(BinaryRBM(zeros(2), zeros(1), fill(0.1, 2, 1)))
    pcd!(
        rbm, binary_data;
        ϵv = 0.04, iters = 1, batchsize = 4, steps = 1,
    )
    @test rbm.scale_v ≈ [0.2, √0.29]
    @test all(isfinite, rbm.visible.par)
    @test all(isfinite, rbm.hidden.par)
    @test all(isfinite, rbm.w)
    @test all(isfinite, free_energy(rbm, binary_data))
end

@testset "exact enumeration of configurations" begin
    #= Scales are positive and bounded away from zero. A near-zero `randn` scale inflates
    the energies, which made a naive log(sum(exp(...))) reference underflow and
    occasionally broke the sampling check below.
    https://github.com/cossio/RestrictedBoltzmannMachines.jl/issues/244 =#
    seed!(244)
    rbm = BinaryStandardizedRBM(
        randn(2), randn(2), randn(2, 2),
        randn(2), randn(2), 0.5 .+ rand(2), 0.5 .+ rand(2)
    )
    vs = generate_sequences(2, 0:1)
    hs = generate_sequences(2, 0:1)

    for v in vs
        @test free_energy(rbm, v) ≈ -logsumexp(-energy(rbm, v, h) for h in hs)
        @test free_energy(rbm, v) ≈ free_energy_v(rbm, v)
    end

    for h in hs
        @test free_energy_h(rbm, h) ≈ -logsumexp(-energy(rbm, v, h) for v in vs)
        @test free_energy_h(rbm, h) ≈ free_energy(mirror(rbm), h)
    end

    sample_v = sample_v_from_v(rbm, bitrand(2, 10000); steps = 10000)
    sample_h = sample_h_from_h(rbm, bitrand(2, 10000); steps = 10000)

    empirical_probs_v = proportionmap(eachcol(sample_v))
    empirical_probs_h = proportionmap(eachcol(sample_h))

    logZ = log_partition(rbm)
    @test logZ ≈ logsumexp(-free_energy_h(rbm, h) for h in hs)
    @test logZ ≈ logsumexp(-free_energy_v(rbm, v) for v in vs)

    exact_probs_v = [exp.(-free_energy_v(rbm, v) .- logZ) for v in vs]
    exact_probs_h = [exp.(-free_energy_h(rbm, h) .- logZ) for h in hs]

    @test vec(exact_probs_v) ≈ vec([get(empirical_probs_v, v, 0.0) for v in vs]) rtol = 0.05
    @test vec(exact_probs_h) ≈ vec([get(empirical_probs_h, h, 0.0) for h in hs]) rtol = 0.05
end

@testset "∂regularize! standardized RBM ($(nameof(typeof(visible))) visible)" for (visible, v) in (
        (Binary(; θ = randn(3)), bitrand(3, 100)),
        (ReLU(; θ = randn(3), γ = rand(3)), rand(3, 100)),
        (dReLU(; θp = randn(3), θn = randn(3), γp = rand(3), γn = rand(3)), randn(3, 100)),
    )
    rbm = StandardizedRBM(visible, Binary(; θ = randn(2)), randn(3, 2), randn(3), randn(2), rand(3), rand(2))
    reg = CompositeRegularizer(
        L2FieldsRegularizer(rand()), L1WeightsRegularizer(rand()),
        L2WeightsRegularizer(rand()), L2L1WeightsRegularizer(rand())
    )

    # on the parameters of the equivalent plain RBM, on the standardized parameters, and mixed
    for regularizer in (
            reg, StandardizedParametersRegularizer(reg),
            CompositeRegularizer(L2WeightsRegularizer(rand()), StandardizedParametersRegularizer(L2FieldsRegularizer(rand()))),
        )
        gs = gradient(rbm) do rbm
            F = mean(free_energy(rbm, v))
            R = regularization_penalty(rbm, regularizer)
            return F + R
        end

        ∂ = ∂free_energy(rbm, v)
        ∂regularize!(∂, rbm, regularizer)

        @test only(gs).visible.par ≈ ∂.visible
        @test only(gs).hidden.par ≈ ∂.hidden
        @test only(gs).w ≈ ∂.w
    end
end

@testset "which parameters a StandardizedRBM regularizes ($(nameof(typeof(visible))) visible)" for visible in (
        Binary(; θ = randn(3)),
        dReLU(; θp = randn(3), θn = randn(3), γp = 1 .+ rand(3), γn = 1 .+ rand(3)),
    )
    rbm = StandardizedRBM(visible, Binary(; θ = randn(2)), randn(3, 2), randn(3), randn(2), 1 .+ rand(3), 1 .+ rand(2))
    λ = rand()
    zero_gradient() = ∂RBM(zero(rbm.visible.par), zero(rbm.hidden.par), zero(rbm.w))
    # closed forms, in the stored parameters: the equivalent plain RBM has the weights
    # w̃ = w / (scale_v ⊗ scale_h), and its visible fields absorb the shift w̃ * offset_h
    scale_w = rbm.scale_v * rbm.scale_h'
    w̃ = rbm.w ./ scale_w
    shift = w̃ * rbm.offset_h
    fields = visible isa dReLU ? (rbm.visible.θp, rbm.visible.θn) : (rbm.visible.θ,)
    nfields = length(fields)
    stack_rows(xs) = reduce(vcat, (x' for x in xs))

    # a bare regularizer penalizes the parameters of the equivalent plain RBM
    @test regularization_penalty(rbm, L2WeightsRegularizer(λ)) ≈ λ / 2 * sum(abs2, w̃)
    @test regularization_penalty(rbm, L2FieldsRegularizer(λ)) ≈ λ / 2 * sum(sum(abs2, θ .- shift) for θ in fields)
    ∂ = ∂regularize!(zero_gradient(), rbm, L2WeightsRegularizer(λ))
    @test ∂.w ≈ λ .* w̃ ./ scale_w
    @test iszero(∂.visible)
    @test iszero(∂.hidden)
    ∂ = ∂regularize!(zero_gradient(), rbm, L2FieldsRegularizer(λ))
    @test ∂.visible[1:nfields, :] ≈ stack_rows(λ .* (θ .- shift) for θ in fields)
    @test iszero(∂.visible[(nfields + 1):end, :])
    @test iszero(∂.hidden)
    # the plain fields absorb the offsets, so the penalty on them also pulls on the weights
    @test ∂.w ≈ -λ .* sum(θ .- shift for θ in fields) * rbm.offset_h' ./ scale_w

    # a wrapped regularizer penalizes the standardized parameters themselves
    @test regularization_penalty(rbm, StandardizedParametersRegularizer(L2WeightsRegularizer(λ))) ≈ λ / 2 * sum(abs2, rbm.w)
    @test regularization_penalty(rbm, StandardizedParametersRegularizer(L2FieldsRegularizer(λ))) ≈ λ / 2 * sum(sum(abs2, θ) for θ in fields)
    ∂ = ∂regularize!(zero_gradient(), rbm, StandardizedParametersRegularizer(L2WeightsRegularizer(λ)))
    @test ∂.w ≈ λ .* rbm.w
    @test iszero(∂.visible)
    @test iszero(∂.hidden)
    ∂ = ∂regularize!(zero_gradient(), rbm, StandardizedParametersRegularizer(L2FieldsRegularizer(λ)))
    @test ∂.visible[1:nfields, :] ≈ stack_rows(λ .* θ for θ in fields)
    @test iszero(∂.visible[(nfields + 1):end, :])
    @test iszero(∂.hidden)
    @test iszero(∂.w)
end

@testset "∂unstandardize ($(nameof(typeof(visible))) visible, $(nameof(typeof(hidden))) hidden)" for visible in (
            Binary(; θ = randn(3, 2)),
            dReLU(; θp = randn(3, 2), θn = randn(3, 2), γp = rand(3, 2), γn = rand(3, 2)),
        ), hidden in (
            Binary(; θ = randn(2)),
            dReLU(; θp = randn(2), θn = randn(2), γp = rand(2), γn = rand(2)),
        )
    rbm = StandardizedRBM(visible, hidden, randn(3, 2, 2), randn(3, 2), randn(2), rand(3, 2), rand(2))
    # a linear function of the parameters of the equivalent plain RBM, with gradient ∂plain
    ∂plain = ∂RBM(randn(size(rbm.visible.par)), randn(size(rbm.hidden.par)), randn(size(rbm.w)))
    f(plain) = sum(∂plain.visible .* plain.visible.par) + sum(∂plain.hidden .* plain.hidden.par) + sum(∂plain.w .* plain.w)
    gs = gradient(rbm -> f(unstandardize(rbm)), rbm)
    ∂ = ∂unstandardize(rbm, ∂plain)
    @test only(gs).visible.par ≈ ∂.visible
    @test only(gs).hidden.par ≈ ∂.hidden
    @test only(gs).w ≈ ∂.w
end

@testset "unstandardized_weights" begin
    rbm = BinaryStandardizedRBM(
        randn(3), randn(2), randn(3, 2),
        randn(3), randn(2), rand(3), rand(2)
    )
    @test unstandardized_weights(rbm) ≈ unstandardize(rbm).w
end

@testset "weight_norms" begin
    rbm = BinaryStandardizedRBM(
        randn(3), randn(2), randn(3, 2),
        randn(3), randn(2), rand(3), rand(2)
    )
    @test weight_norms(rbm) ≈ weight_norms(unstandardize(rbm))
end

@testset "rescale_weights! std $(nameof(typeof(hidden)))" for hidden in (
        ReLU(; θ = randn(2), γ = 0.1 .+ rand(2)),
        dReLU(; θp = randn(2), θn = randn(2), γp = rand(2), γn = rand(2)),
    )
    rbm = StandardizedRBM(Binary(; θ = randn(3)), hidden, randn(3, 2), randn(3), randn(2), rand(3), rand(2))
    rbm_copy = deepcopy(rbm)

    @test @inferred rescale_weights!(rbm)
    @test weight_norms(unstandardize(rbm)) ≈ ones(size(rbm.hidden))

    v = bitrand(size(rbm.visible)..., 100)
    @test free_energy(rbm, v) ≈ free_energy(rbm_copy, v) .- sum(log, weight_norms(rbm_copy))
end

@testset "rescale_weights! std preserves zero-norm hidden units" begin
    rbm = StandardizedRBM(
        Binary(; θ = randn(2)),
        ReLU(; θ = [0.25, -0.5], γ = [1.5, 2.0]),
        [0.0 6.0; 0.0 8.0],
        randn(2),
        [0.75, -0.25],
        [2.0, 4.0],
        [1.5, 2.0],
    )
    rbm0 = deepcopy(rbm)
    v = Bool[0 0 1 1; 0 1 0 1]
    F0 = free_energy(rbm, v)
    ω = weight_norms(rbm)
    @test ω[1] == 0
    @test ω[2] > 0

    @test @inferred rescale_weights!(rbm)
    @test weight_norms(rbm) ≈ [0, 1]
    @test rbm.w == rbm0.w
    @test rbm.hidden.par[:, 1] == rbm0.hidden.par[:, 1]
    @test rbm.offset_h[1] == rbm0.offset_h[1]
    @test rbm.scale_h[1] == rbm0.scale_h[1]
    @test all(isfinite, rbm.w)
    @test all(isfinite, rbm.hidden.par)
    @test all(isfinite, rbm.offset_h)
    @test all(isfinite, rbm.scale_h)
    @test free_energy(rbm, v) ≈ F0 .- log(ω[2])
end

@testset "rescale_weights! std Binary" begin
    rbm = BinaryStandardizedRBM(
        randn(3), randn(2), randn(3, 2),
        randn(3), randn(2), rand(3), rand(2)
    )
    rbm_copy = deepcopy(rbm)

    @test @inferred !rescale_weights!(rbm)

    v = bitrand(size(rbm.visible)..., 100)
    @test free_energy(rbm, v) ≈ free_energy(rbm_copy, v)
end

using Random: rand!
using RestrictedBoltzmannMachines: SpinRBM, Spin, PottsGumbel, potts_to_gumbel, gumbel_to_potts,
    log_pseudolikelihood

@testset "BinaryStandardizedRBM / SpinStandardizedRBM constructors" begin
    a, b, w = randn(3), randn(2), randn(3, 2)
    offset_v, offset_h = randn(3), randn(2)
    scale_v, scale_h = 1 .+ rand(3), 1 .+ rand(2)
    for (cons, base) in ((BinaryStandardizedRBM, BinaryRBM), (SpinStandardizedRBM, SpinRBM))
        srbm = @inferred cons(a, b, w, offset_v, offset_h, scale_v, scale_h)
        @test srbm.visible.θ == a
        @test srbm.hidden.θ == b
        @test srbm.w == w
        @test srbm.offset_v == offset_v
        @test srbm.offset_h == offset_h
        @test srbm.scale_v == scale_v
        @test srbm.scale_h == scale_h

        # 3-argument variant has trivial offsets and scales, equivalent to the plain RBM
        srbm0 = @inferred cons(a, b, w)
        @test iszero(srbm0.offset_v)
        @test iszero(srbm0.offset_h)
        @test all(isone, srbm0.scale_v)
        @test all(isone, srbm0.scale_h)
        v = base === BinaryRBM ? bitrand(3, 5) : rand((-1, 1), 3, 5)
        h = base === BinaryRBM ? bitrand(2, 5) : rand((-1, 1), 2, 5)
        @test energy(srbm0, v, h) ≈ energy(base(a, b, w), v, h)
    end
end

@testset "standardize_visible / standardize_hidden" begin
    rbm = BinaryRBM(randn(3), randn(2), randn(3, 2))
    offset_v, scale_v = randn(3), 1 .+ rand(3)
    offset_h, scale_h = randn(2), 1 .+ rand(2)
    v = bitrand(3, 7)
    F0 = free_energy(rbm, v)

    # from a plain RBM, with target offsets/scales
    srbm_v = @inferred standardize_visible(rbm, offset_v, scale_v)
    srbm_h = @inferred standardize_hidden(rbm, offset_h, scale_h)
    @test srbm_v.offset_v == offset_v
    @test srbm_v.scale_v == scale_v
    @test iszero(srbm_v.offset_h)
    @test srbm_h.offset_h == offset_h
    @test srbm_h.scale_h == scale_h
    @test iszero(srbm_h.offset_v)
    # standardization is a gauge transformation: free energies shift by a constant
    for srbm in (srbm_v, srbm_h)
        F = free_energy(srbm, v)
        @test F ≈ F0 .+ mean(F - F0)
    end

    # no-argument variants reset the offsets/scales of one side
    srbm = standardize(rbm)
    srbm.offset_v .= randn(3)
    srbm.scale_v .= 1 .+ rand(3)
    srbm.offset_h .= randn(2)
    srbm.scale_h .= 1 .+ rand(2)
    F1 = free_energy(srbm, v)
    rv = @inferred standardize_visible(srbm)
    @test iszero(rv.offset_v)
    @test all(isone, rv.scale_v)
    @test rv.offset_h == srbm.offset_h
    @test rv.scale_h == srbm.scale_h
    rh = @inferred standardize_hidden(srbm)
    @test iszero(rh.offset_h)
    @test all(isone, rh.scale_h)
    @test rh.offset_v == srbm.offset_v
    @test rh.scale_v == srbm.scale_v
    for srbm_reset in (rv, rh)
        F = free_energy(srbm_reset, v)
        @test F ≈ F1 .+ mean(F - F1)
    end

    # plain-RBM no-offset variants are equivalent to standardize
    for srbm_plain in (standardize_visible(rbm), standardize_hidden(rbm))
        @test srbm_plain isa StandardizedRBM
        @test free_energy(srbm_plain, v) ≈ F0
    end
end

@testset "potts_to_gumbel / gumbel_to_potts StandardizedRBM" begin
    q = 3
    rbm = RBM(Potts(; θ = randn(q, 2)), Binary(; θ = randn(2)), randn(q, 2, 2))
    srbm = standardize(rbm)
    rand!(srbm.offset_v)
    rand!(srbm.offset_h)
    srbm.scale_v .= 1 .+ rand(q, 2)
    srbm.scale_h .= 1 .+ rand(2)

    grbm = potts_to_gumbel(srbm)
    @test grbm isa StandardizedRBM
    @test grbm.visible isa PottsGumbel
    @test grbm.visible.par == srbm.visible.par
    @test grbm.hidden.par == srbm.hidden.par
    @test grbm.w == srbm.w
    @test grbm.offset_v == srbm.offset_v
    @test grbm.offset_h == srbm.offset_h
    @test grbm.scale_v == srbm.scale_v
    @test grbm.scale_h == srbm.scale_h

    back = gumbel_to_potts(grbm)
    @test back.visible isa Potts
    @test back.visible.par == srbm.visible.par
    @test back.w == srbm.w
end

@testset "log_pseudolikelihood of StandardizedRBM" begin
    # With a single visible site the stochastic estimator is deterministic,
    # and standardization must not change the pseudolikelihood.
    srbm = BinaryStandardizedRBM(randn(1), randn(2), randn(1, 2), randn(1), randn(2), 1 .+ rand(1), 1 .+ rand(2))
    v = bitrand(1, 7)
    @test log_pseudolikelihood(srbm, v) ≈ log_pseudolikelihood(unstandardize(srbm), v)
    @test log_pseudolikelihood(srbm, v; exact = true) ≈
        log_pseudolikelihood(unstandardize(srbm), v; exact = true)
end

@testset "regularization_penalty StandardizedRBM" begin
    srbm = BinaryStandardizedRBM(randn(3), randn(2), randn(3, 2), randn(3), randn(2), 1 .+ rand(3), 1 .+ rand(2))
    reg = CompositeRegularizer(
        L1WeightsRegularizer(0.1), L2WeightsRegularizer(0.2), L2L1WeightsRegularizer(0.3), L2FieldsRegularizer(0.4)
    )
    @test regularization_penalty(srbm, reg) ≈ regularization_penalty(unstandardize(srbm), reg)
    @test regularization_penalty(srbm, StandardizedParametersRegularizer(reg)) ≈
        regularization_penalty(RBM(srbm.visible, srbm.hidden, srbm.w), reg)
end
