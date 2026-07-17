using LinearAlgebra: I, norm
using Random: bitrand
using RestrictedBoltzmannMachines: ∂free_energy, ∂free_energy_h, ∂free_energy_v, ∂regularize!
using RestrictedBoltzmannMachines: Binary, Spin, dReLU, Gaussian, Potts, ReLU, nsReLU, pReLU, xReLU
using RestrictedBoltzmannMachines: RBM, StandardizedRBM
using RestrictedBoltzmannMachines: BinaryRBM, BinaryStandardizedRBM, SpinStandardizedRBM
using RestrictedBoltzmannMachines: weight_norms, delta_energy
using RestrictedBoltzmannMachines: rescale_weights!, rescale_hidden_activations!
using RestrictedBoltzmannMachines: energy, interaction_energy, free_energy, free_energy_h, free_energy_v
using RestrictedBoltzmannMachines: generate_sequences
using RestrictedBoltzmannMachines: inputs_h_from_v, inputs_v_from_h
using RestrictedBoltzmannMachines: log_partition
using RestrictedBoltzmannMachines: mean_h_from_v, mean_v_from_h, var_h_from_v, var_v_from_h
using RestrictedBoltzmannMachines: mirror
using RestrictedBoltzmannMachines: pcd!, regularization_penalty
using RestrictedBoltzmannMachines: sample_h_from_h, sample_v_from_v, sample_from_inputs
using RestrictedBoltzmannMachines: shift_fields, shift_fields!
using RestrictedBoltzmannMachines: standardize, unstandardize, standardize!, unstandardized_weights
using RestrictedBoltzmannMachines: standardize_hidden, standardize_visible
using StatsBase: proportionmap
using Statistics: mean
using Test: @inferred, @test, @testset
using Zygote: gradient

function energy_shift(offset::AbstractArray, x::AbstractArray)
    @assert size(offset) == size(x)[1:ndims(offset)]
    if ndims(offset) == ndims(x)
        return -sum(offset .* x)
    elseif ndims(offset) < ndims(x)
        ΔE = -sum(offset .* x; dims=1:ndims(offset))
        return reshape(ΔE, size(x)[(ndims(offset) + 1):end])
    end
end

@testset "shift_fields" begin
    N = (3, 4)
    layers = (
        Binary(; θ = randn(N...)),
        Spin(; θ = randn(N...)),
        Potts(; θ = randn(N...)),
        Gaussian(; θ = randn(N...), γ = rand(N...)),
        ReLU(; θ = randn(N...), γ = rand(N...)),
        dReLU(; θp = randn(N...), θn = randn(N...), γp = rand(N...), γn = rand(N...)),
        pReLU(; θ = randn(N...), γ = rand(N...), Δ = randn(N...), η = rand(N...) .- 0.5),
        xReLU(; θ = randn(N...), γ = rand(N...), Δ = randn(N...), ξ = randn(N...)),
    )
    for layer in layers
        offset = randn(size(layer)...)
        x = sample_from_inputs(layer, randn(size(layer)..., 2, 3))
        layer_shifted = @inferred shift_fields(layer, offset)
        @test energy(layer_shifted, x) ≈ energy(layer, x) + energy_shift(offset, x)
    end
end

@testset "shift_fields!" begin
    N = (3, 4)
    layers = (
        Binary(; θ = randn(N...)),
        Spin(; θ = randn(N...)),
        Potts(; θ = randn(N...)),
        Gaussian(; θ = randn(N...), γ = rand(N...)),
        ReLU(; θ = randn(N...), γ = rand(N...)),
        dReLU(; θp = randn(N...), θn = randn(N...), γp = rand(N...), γn = rand(N...)),
        pReLU(; θ = randn(N...), γ = rand(N...), Δ = randn(N...), η = rand(N...) .- 0.5),
        xReLU(; θ = randn(N...), γ = rand(N...), Δ = randn(N...), ξ = randn(N...)),
    )
    for layer in layers
        offset = randn(size(layer)...)
        x = sample_from_inputs(layer, randn(size(layer)..., 2, 3))
        E = energy(layer, x)
        @inferred shift_fields!(layer, offset)
        @test energy(layer, x) ≈ E + energy_shift(offset, x)
    end
end

@testset "standardize" begin
    rbm = BinaryRBM(randn(3), randn(2), randn(3,2))
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
    std_rbm = @inferred BinaryStandardizedRBM(randn(3), randn(2), randn(3,2), randn(3), randn(2), rand(3), rand(2))
    rbm = @inferred unstandardize(std_rbm)
    @test rbm isa RBM
    @test unstandardize(rbm) == rbm
    v = bitrand(3, 10)
    h = bitrand(2, 10)
    @test energy(std_rbm, v, h) .- delta_energy(std_rbm) ≈ energy(rbm, v, h)
end

@testset "delta_energy" begin
    rbm = BinaryRBM(randn(3), randn(2), randn(3,2))
    std_rbm = @inferred standardize(rbm, randn(3), randn(2), rand(3), rand(2))
    @test iszero(@inferred delta_energy(rbm))
    @test iszero(delta_energy(standardize(std_rbm)))
    @test delta_energy(rbm) isa Real
    @test delta_energy(std_rbm) isa Real
end

@testset "standardize!" begin
    rbm = @inferred BinaryStandardizedRBM(
        randn(3), randn(2), randn(3,2),
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

@testset "∂free energy" begin
    rbm = @inferred BinaryStandardizedRBM(
        randn(3), randn(2), randn(3,2),
        randn(3), randn(2), rand(3), rand(2)
    )
    v = bitrand(size(rbm.visible)..., 10)
    h = bitrand(size(rbm.hidden)..., 10)

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

@testset "∂free_energy nsReLU" begin
    rbm = standardize(
        RBM(Binary(; θ=randn(3)), nsReLU(; θ=randn(2), ξ=randn(2), Δ=randn(2)), randn(3,2)),
        randn(3), randn(2), 0.1 .+ rand(3), 0.1 .+ rand(2)
    )
    v = bitrand(size(rbm.visible)..., 10)
    h = randn(size(rbm.hidden)..., 10)

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
        RBM(Binary(; θ=randn(3)), ReLU(; θ=randn(2), γ=0.1 .+ rand(2)), randn(3,2)),
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
    @test var_h_from_v(rbm, v) ≈ h_var ./ λ.^2
    @test mean_v_from_h(rbm, h_ave ./ λ) ≈ v_ave
    @test var_v_from_h(rbm, h_ave ./ λ) ≈ v_var
    @test all(rbm.scale_h .≈ 1)
    @test free_energy(rbm, v) ≈ F .+ sum(log, λ)
end

@testset "standardized pcd" begin
    rbm = standardize(RBM(Spin(; θ=zeros(10)), Spin(; θ=zeros(7)), randn(10, 7) / √10))
    @test iszero(rbm.visible.θ) && iszero(rbm.hidden.θ)

    data = ones(10, 4)
    data[1:3, 2] .= -1
    data[:, 3:4] .= -data[:, 1:2] # ensure data has zero mean
    @test iszero(mean(data; dims=2))

    state, ps = pcd!(
        rbm, data;
        ps = (; w=rbm.w), # train only weights
        steps=10, batchsize=4, iters=1000,
        ϵv=1f-1, ϵh=0f0, damping=1f-1
    )

    # The fields are not exactly zero because centering introduces minor numerical fluctuations.
    @test norm(rbm.visible.θ) < 1e-13
    @test iszero(rbm.hidden.θ)
end

@testset "exact enumeration of configurations" begin
    rbm = BinaryStandardizedRBM(
        randn(2), randn(2), randn(2,2),
        randn(2), randn(2), randn(2), randn(2)
    )
    vs = generate_sequences(2, 0:1)
    hs = generate_sequences(2, 0:1)

    for v = vs
        @test free_energy(rbm, v) ≈ -log(sum(exp(-energy(rbm, v, h)) for h = hs))
        @test free_energy(rbm, v) ≈ free_energy_v(rbm, v)
    end

    for h = hs
        @test free_energy_h(rbm, h) ≈ -log(sum(exp(-energy(rbm, v, h)) for v = vs))
        @test free_energy_h(rbm, h) ≈ free_energy(mirror(rbm), h)
    end

    sample_v = sample_v_from_v(rbm, bitrand(2, 10000); steps=10000)
    sample_h = sample_h_from_h(rbm, bitrand(2, 10000); steps=10000)

    empirical_probs_v = proportionmap(eachcol(sample_v))
    empirical_probs_h = proportionmap(eachcol(sample_h))

    logZ = log_partition(rbm)
    @test logZ ≈ log(sum(exp(-free_energy_h(rbm, h)) for h = hs))
    @test logZ ≈ log(sum(exp(-free_energy_v(rbm, v)) for v = vs))

    exact_probs_v = [exp.(-free_energy_v(rbm, v) .- logZ) for v = vs]
    exact_probs_h = [exp.(-free_energy_h(rbm, h) .- logZ) for h = hs]

    @test vec(exact_probs_v) ≈ vec([get(empirical_probs_v, v, 0.) for v = vs]) rtol=0.05
    @test vec(exact_probs_h) ≈ vec([get(empirical_probs_h, h, 0.) for h = hs]) rtol=0.05
end

@testset "∂regularize! standardized RBM with Binary" begin
    rbm = BinaryStandardizedRBM(
        randn(3), randn(2), randn(3,2),
        randn(3), randn(2), rand(3), rand(2)
    )
    v = bitrand(3, 100)
    vdims = ntuple(identity, ndims(rbm.visible))
    N = length(rbm.visible)

    l2_fields = rand()
    l1_weights = rand()
    l2_weights = rand()
    l2l1_weights = rand()

    for regularize_unstandardized = (false, true)
        gs = gradient(rbm) do rbm
            F = mean(free_energy(rbm, v))
            R = regularization_penalty(rbm; regularize_unstandardized, l1_weights, l2_weights, l2l1_weights, l2_fields)
            return F + R
        end

        ∂ = ∂free_energy(rbm, v)
        ∂regularize!(∂, rbm; regularize_unstandardized, l2_fields, l1_weights, l2_weights, l2l1_weights)

        @test only(gs).visible.par ≈ ∂.visible
        @test only(gs).hidden.par ≈ ∂.hidden
        @test only(gs).w ≈ ∂.w
    end
end

@testset "∂regularize! standardized RBM with ReLU" begin
    rbm = StandardizedRBM(
        ReLU(; θ=randn(3), γ=rand(3)),
        Binary(; θ=randn(2)),
        randn(3,2),
        randn(3), randn(2), rand(3), rand(2)
    )
    v = rand(3, 100)
    vdims = ntuple(identity, ndims(rbm.visible))
    N = length(rbm.visible)

    l2_fields = rand()
    l1_weights = rand()
    l2_weights = rand()
    l2l1_weights = rand()

    for regularize_unstandardized = (false, true)
        gs = gradient(rbm) do rbm
            F = mean(free_energy(rbm, v))
            R = regularization_penalty(rbm; regularize_unstandardized, l1_weights, l2_weights, l2l1_weights, l2_fields)
            return F + R
        end

        ∂ = ∂free_energy(rbm, v)
        ∂regularize!(∂, rbm; regularize_unstandardized, l2_fields, l1_weights, l2_weights, l2l1_weights)

        @test only(gs).visible.par ≈ ∂.visible
        @test only(gs).hidden.par ≈ ∂.hidden
        @test only(gs).w ≈ ∂.w
    end
end

@testset "∂regularize! standardized RBM with dReLU" begin
    rbm = StandardizedRBM(
        dReLU(; θp=randn(3), θn=randn(3), γp=rand(3), γn=rand(3)),
        Binary(; θ=randn(2)),
        randn(3,2),
        randn(3), randn(2), rand(3), rand(2)
    )
    v = randn(3, 100)
    vdims = ntuple(identity, ndims(rbm.visible))
    N = length(rbm.visible)

    l2_fields = rand()
    l1_weights = rand()
    l2_weights = rand()
    l2l1_weights = rand()

    for regularize_unstandardized = (false, true)
        gs = gradient(rbm) do rbm
            F = mean(free_energy(rbm, v))
            R = regularization_penalty(rbm; regularize_unstandardized, l1_weights, l2_weights, l2l1_weights, l2_fields)
            return F + R
        end

        ∂ = ∂free_energy(rbm, v)
        ∂regularize!(∂, rbm; regularize_unstandardized, l2_fields, l1_weights, l2_weights, l2l1_weights)

        @test only(gs).visible.par ≈ ∂.visible
        @test only(gs).hidden.par ≈ ∂.hidden
        @test only(gs).w ≈ ∂.w
    end
end

@testset "unstandardized_weights" begin
    rbm = BinaryStandardizedRBM(
        randn(3), randn(2), randn(3,2),
        randn(3), randn(2), rand(3), rand(2)
    )
    @test unstandardized_weights(rbm) ≈ unstandardize(rbm).w
end

@testset "weight_norms" begin
    rbm = BinaryStandardizedRBM(
        randn(3), randn(2), randn(3,2),
        randn(3), randn(2), rand(3), rand(2)
    )
    @test weight_norms(rbm) ≈ weight_norms(unstandardize(rbm))
end

@testset "rescale_weights! std ReLU" begin
    rbm = StandardizedRBM(
        Binary(; θ=randn(3)), ReLU(; θ=randn(2), γ=0.1 .+ rand(2)), randn(3,2),
        randn(3), randn(2), rand(3), rand(2)
    )
    rbm_copy = deepcopy(rbm)

    @test @inferred rescale_weights!(rbm)
    @test weight_norms(unstandardize(rbm)) ≈ ones(size(rbm.hidden))

    v = bitrand(size(rbm.visible)..., 100)
    @test free_energy(rbm, v) ≈ free_energy(rbm_copy, v) .- sum(log, weight_norms(rbm_copy))
end

@testset "rescale_weights! std dReLU" begin
    rbm = StandardizedRBM(
        Binary(; θ=randn(3)), dReLU(; θp=randn(2), θn=randn(2), γp=rand(2), γn=rand(2)), randn(3,2),
        randn(3), randn(2), rand(3), rand(2)
    )
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
        randn(3), randn(2), randn(3,2),
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
end

@testset "regularization_penalty StandardizedRBM" begin
    srbm = BinaryStandardizedRBM(randn(3), randn(2), randn(3, 2), randn(3), randn(2), 1 .+ rand(3), 1 .+ rand(2))
    kw = (; l1_weights = 0.1, l2_weights = 0.2, l2l1_weights = 0.3, l2_fields = 0.4)
    @test regularization_penalty(srbm; kw...) ≈ regularization_penalty(unstandardize(srbm); kw...)
    @test regularization_penalty(srbm; regularize_unstandardized = false, kw...) ≈
        regularization_penalty(RBM(srbm.visible, srbm.hidden, srbm.w); kw...)
end
