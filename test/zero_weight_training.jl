using Test: @test, @testset, @test_throws
using Random: seed!
using Statistics: mean
using FillArrays: Ones
import Optimisers
import RestrictedBoltzmannMachines as RBMs
using RestrictedBoltzmannMachines: RBM, Binary, Gaussian, BinaryRBM,
    CenteredRBM, StandardizedRBM, center, standardize, pcd!, initialize!

struct CountingDescent{T, R} <: Optimisers.AbstractRule
    eta::T
    calls::R
end

Optimisers.init(::CountingDescent, x::AbstractArray) = nothing

function Optimisers.apply!(rule::CountingDescent, state, x, dx)
    rule.calls[] += 1
    return state, rule.eta .* dx
end

function base_rbm()
    return BinaryRBM([0.1, -0.2], [0.05], reshape([0.2, -0.1], 2, 1))
end

wrap_rbm(::Val{:plain}, rbm::RBM) = rbm
wrap_rbm(::Val{:centered}, rbm::RBM) = center(rbm)
wrap_rbm(::Val{:standardized}, rbm::RBM) = standardize(rbm)

function model_state(rbm::RBM)
    return (; visible = copy(rbm.visible.par), hidden = copy(rbm.hidden.par), w = copy(rbm.w))
end

function model_state(rbm::CenteredRBM)
    return (;
        visible = copy(rbm.visible.par),
        hidden = copy(rbm.hidden.par),
        w = copy(rbm.w),
        offset_v = copy(rbm.offset_v),
        offset_h = copy(rbm.offset_h),
    )
end

function model_state(rbm::StandardizedRBM)
    return (;
        visible = copy(rbm.visible.par),
        hidden = copy(rbm.hidden.par),
        w = copy(rbm.w),
        offset_v = copy(rbm.offset_v),
        offset_h = copy(rbm.offset_h),
        scale_v = copy(rbm.scale_v),
        scale_h = copy(rbm.scale_h),
    )
end

all_finite(rbm) = all(x -> all(isfinite, x), values(model_state(rbm)))

@testset "invalid training weights" begin
    data = zeros(2, 2)
    for bad_wts in (
            [2.0, -1.0],
            [1.0, 0.0], # zero weights are rejected: drop such samples beforehand
            [1.0, NaN],
            [1.0, Inf],
            ComplexF64[1, 1], # complex weights are not real
        )
        @test_throws ArgumentError RBMs._validate_weights(bad_wts)
        rbm = base_rbm()
        before = model_state(rbm)
        @test_throws ArgumentError pcd!(
            rbm, data;
            wts = bad_wts, batchsize = 1, iters = 0,
            zerosum = false, rescale = false,
        )
        @test model_state(rbm) == before
    end
end

@testset "wmean is a plain weighted mean" begin
    @test RBMs.wmean([1.0, 3.0]; wts = [1.0, 3.0]) ≈ 2.5
    @test RBMs.wmean([1.0, 3.0]; wts = Any[1.0, 3.0]) ≈ 2.5
    @test RBMs.wmean([1.0, 3.0]; wts = fill(big"1e400", 2)) ≈ 2.0
    # the kernel does not validate weights (the training entry points reject
    # non-positive weights): zero weights annihilate finite samples
    # arithmetically, but do not mask non-finite samples
    @test RBMs.wmean([1.0, 2.0, 4.0]; wts = [0.0, 1.0, 3.0]) ≈ 3.5
    @test isnan(RBMs.wmean([NaN, 2.0]; wts = [0.0, 1.0]))

    # lazy uniform weights reduce like a plain mean, without promoting —
    # on partial and full (default) reductions alike
    A = rand(Float32, 2, 5)
    @test RBMs.wmean(A; wts = Ones{Bool}(5)) ≈ vec(mean(A; dims = 2))
    @test RBMs.wmean(A; wts = Ones{Bool}(5)) isa Vector{Float32}
    @test RBMs.wmean(A) ≈ mean(A)
    @test RBMs.wmean(A) isa Float32
    @test RBMs.wmean(Float32[1, 2]) isa Float32

    # integer data accumulates in float on the uniform fast path too,
    # matching `mean` instead of wrapping like an integer `sum`
    big_ints = [typemax(Int), typemax(Int)]
    @test RBMs.wmean(big_ints) ≈ mean(big_ints) ≈ float(typemax(Int))
    @test RBMs.wmean(reshape(big_ints, 1, 2); wts = Ones{Bool}(2)) ≈ [float(typemax(Int))]
end

@testset "training weight checks" begin
    # lazy uniform weights must still be real-valued to skip validation
    @test RBMs._validate_weights(Ones{Bool}(3)) isa Ones
    @test_throws ArgumentError RBMs._validate_weights(Ones{ComplexF64}(2))

    # Empty data and undersized weights are rejected before mutation. The
    # default `moments` kwarg already fails computing statistics of such
    # inputs; explicit `moments` routes to the training-loop checks.
    empty_rbm = base_rbm()
    before = model_state(empty_rbm)
    moments = RBMs.moments_from_samples(empty_rbm.visible, [0.0 1.0; 1.0 0.0])
    @test_throws ArgumentError pcd!(
        empty_rbm, zeros(2, 0);
        moments, batchsize = 1, iters = 0, zerosum = false, rescale = false,
    )
    @test model_state(empty_rbm) == before
    @test_throws DimensionMismatch pcd!(
        base_rbm(), zeros(2, 3);
        wts = [1.0, 1.0], moments,
        batchsize = 1, iters = 0, zerosum = false, rescale = false,
    )

    # batchsize > nsamples clamps to one full batch instead of silently
    # performing zero training iterations
    rbm = base_rbm()
    updates = Ref(0)
    pcd!(
        rbm, [0.0 1.0 1.0; 1.0 0.0 1.0];
        batchsize = 5, iters = 2, steps = 1,
        optim = CountingDescent(0.01, updates),
        zerosum = false, rescale = false,
    )
    @test updates[] > 0
    @test all_finite(rbm)
end

@testset "explicitly passed moments are used as given" begin
    rbm = base_rbm()
    data = [0.0 1.0; 1.0 0.0]
    moments = RBMs.moments_from_samples(rbm.visible, data)
    pcd!(rbm, data; batchsize = 2, iters = 1, moments, zerosum = false, rescale = false)
    @test all_finite(rbm)
end

@testset "initialize! weight checks" begin
    data = [1.0 0.0 1.0; 0.0 1.0 1.0]

    # the default weights are lazy uniform `Ones`, like everywhere else
    rbm = base_rbm()
    initialize!(rbm, data)
    @test all_finite(rbm)
    layer = RBMs.Binary((2,))
    initialize!(layer, data; wts = Ones{Bool}(3))
    @test all(isfinite, layer.par)

    # weights enter the reductions unrescaled; only relative values matter
    # for benign scales
    scaled_rbm = base_rbm()
    unit_rbm = base_rbm()
    seed!(303)
    initialize!(scaled_rbm, data; wts = fill(2.0, 3))
    seed!(303)
    initialize!(unit_rbm, data; wts = ones(3))
    @test all_finite(scaled_rbm)
    @test model_state(scaled_rbm) == model_state(unit_rbm)

    # invalid weights fail loudly
    @test_throws ArgumentError initialize!(base_rbm(), data; wts = [1.0, -1.0, 1.0])
    @test_throws ArgumentError initialize!(base_rbm(), data; wts = [1.0, 0.0, 1.0])
    @test_throws ArgumentError initialize!(base_rbm(), data; wts = [1.0, NaN, 1.0])
    # undersized weights are rejected instead of silently truncating the data
    @test_throws DimensionMismatch initialize!(base_rbm(), data; wts = [1.0, 1.0])

    # empty data fails before mutating parameters to NaN
    empty_rbm = base_rbm()
    before = model_state(empty_rbm)
    @test_throws ArgumentError initialize!(empty_rbm, zeros(2, 0))
    @test model_state(empty_rbm) == before

    # narrow integer data and weights are floated before the reductions
    int8_rbm = base_rbm()
    seed!(404)
    RBMs.initialize_w!(int8_rbm, reshape(Int8[2, 2], 2, 1); wts = Int8[100])
    float_rbm = base_rbm()
    seed!(404)
    RBMs.initialize_w!(float_rbm, reshape([2.0, 2.0], 2, 1); wts = [100.0])
    @test int8_rbm.w == float_rbm.w
    @test all(isfinite, int8_rbm.w)
end

@testset "∂free_energy with zero weights on finite samples" begin
    # the kernel does not validate weights (the training entry points reject
    # non-positive weights); zero weights contribute exactly zero for finite data
    rbm = base_rbm()
    v = [
        1.0 0.0 1.0
        0.0 1.0 0.0
    ]
    wts = [0.0, 1.0, 2.0]
    ∂ = RBMs.∂free_energy(rbm, v; wts)
    ∂ref = RBMs.∂free_energy(rbm, v[:, 2:3]; wts = wts[2:3])
    @test ∂.visible ≈ ∂ref.visible
    @test ∂.hidden ≈ ∂ref.hidden
    @test ∂.w ≈ ∂ref.w
end

function mutation_sensitive_rbm(::Val{:plain})
    return RBM(
        Binary(; θ = [0.1, -0.2]),
        Gaussian(; θ = [0.3], γ = [2.0]),
        reshape([3.0, 4.0], 2, 1),
    )
end

mutation_sensitive_rbm(kind) = wrap_rbm(kind, base_rbm())

function check_all_zero_pcd(kind)
    data = [0.0 1.0; 1.0 0.0]
    wts = zeros(2)
    rbm = mutation_sensitive_rbm(kind)
    before = model_state(rbm)
    updates = Ref(0)
    callbacks = Ref(0)

    @test_throws ArgumentError pcd!(
        rbm, data;
        wts, batchsize = 2, iters = 1, steps = 1,
        optim = CountingDescent(0.01, updates),
        callback = (; kwargs...) -> (callbacks[] += 1),
    )
    @test model_state(rbm) == before
    @test iszero(updates[])
    @test iszero(callbacks[])
    return nothing
end

@testset "invalid batchsize fails before mutation" begin
    rbm = mutation_sensitive_rbm(Val(:plain))
    before = model_state(rbm)
    @test_throws ArgumentError pcd!(rbm, [0.0 1.0; 1.0 0.0]; batchsize = 0)
    @test model_state(rbm) == before
end

@testset "all-zero weights fail before mutation ($name PCD)" for (name, kind) in [
        ("plain", Val(:plain)),
        ("centered", Val(:centered)),
        ("standardized", Val(:standardized)),
    ]
    check_all_zero_pcd(kind)
end
