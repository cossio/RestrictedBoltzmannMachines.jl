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

function callback_log()
    iterations = Int[]
    weights = Any[]
    callback = (; iter, wd, kwargs...) -> begin
        push!(iterations, iter)
        push!(weights, copy(wd))
        return nothing
    end
    return (; callback, iterations, weights)
end

function weighted_data()
    # The NaNs are deliberately attached only to zero-weight observations. If
    # zero weights are truly ignored, none of the training paths ever sees them.
    data = [
        NaN NaN 0.0 1.0
        NaN NaN 1.0 0.0
    ]
    wts = [0.0, 0.0, 1.0, 2.0]
    return data, wts
end

function train_pcd!(
        ::Val{:plain}, rbm, data, wts, vm, optim, callback;
        iters::Int, batchsize::Int,
    )
    return pcd!(
        rbm, data;
        wts, vm, optim, callback, iters, batchsize,
        steps = 1, shuffle = false, zerosum = false, rescale = false,
    )
end

function train_pcd!(
        ::Val{:centered}, rbm, data, wts, vm, optim, callback;
        iters::Int, batchsize::Int,
    )
    return pcd!(
        rbm, data;
        wts, vm, optim, callback, iters, batchsize,
        steps = 1, hidden_offset_damping = 1 // 4,
        zerosum = false, rescale = false,
    )
end

function train_pcd!(
        ::Val{:standardized}, rbm, data, wts, vm, optim, callback;
        iters::Int, batchsize::Int,
    )
    return pcd!(
        rbm, data;
        wts, vm, optim, callback, iters, batchsize,
        steps = 1, shuffle = false, damping = 1 // 4, ϵv = 0.1, ϵh = 0.1,
        zerosum = false, rescale_hidden = false,
    )
end

function check_pcd_filter_equivalence(kind, seed)
    data, wts = weighted_data()
    positive = findall(!iszero, wts)
    filtered_data = data[:, positive]
    filtered_wts = wts[positive]

    initial = base_rbm()
    mixed_rbm = wrap_rbm(kind, deepcopy(initial))
    filtered_rbm = wrap_rbm(kind, deepcopy(initial))
    mixed_vm = falses(2, 2)
    filtered_vm = copy(mixed_vm)

    mixed_calls = Ref(0)
    filtered_calls = Ref(0)
    mixed_log = callback_log()

    seed!(seed)
    train_pcd!(
        kind, mixed_rbm, data, wts, mixed_vm,
        CountingDescent(0.01, mixed_calls), mixed_log.callback;
        iters = 2, batchsize = 2,
    )
    seed!(seed)
    train_pcd!(
        kind, filtered_rbm, filtered_data, filtered_wts, filtered_vm,
        CountingDescent(0.01, filtered_calls), Returns(nothing);
        iters = 2, batchsize = 2,
    )

    @test mixed_log.iterations == 1:2
    @test length(mixed_log.weights) == 2
    @test all(wd -> all(w -> w > 0, wd), mixed_log.weights)
    @test mixed_calls[] == filtered_calls[] == 3 * 2
    @test model_state(mixed_rbm) == model_state(filtered_rbm)
    @test mixed_vm == filtered_vm
    @test all_finite(mixed_rbm)
    return nothing
end

@testset "zero-weight PCD matches filtering ($name)" for (name, kind, seed) in [
        ("plain", Val(:plain), 101),
        ("centered", Val(:centered), 102),
        ("standardized", Val(:standardized), 103),
    ]
    check_pcd_filter_equivalence(kind, seed)
end

@testset "invalid training weights" begin
    data = zeros(2, 2)
    for bad_wts in (
            [2.0, -1.0],
            [1.0, NaN],
            [1.0, Inf],
            ComplexF64[1, 1], # complex weights are not real
        )
        @test_throws ArgumentError RBMs._prepare_training_data(data, bad_wts; batchsize = 1)
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

@testset "finite extreme $weight_type weights are scale-stable ($name PCD)" for
    (name, kind, seed) in [
            ("plain", Val(:plain), 109),
            ("centered", Val(:centered), 110),
            ("standardized", Val(:standardized), 111),
        ],
        (weight_type, extreme_weight) in [
            ("Float64", floatmax(Float64)),
            ("UInt128", typemax(UInt128)),
        ]
    data = [
        NaN 0.0 1.0
        NaN 1.0 0.0
    ]
    extreme_wts = [zero(extreme_weight), extreme_weight, extreme_weight]
    unit_wts = [0.0, 1.0, 1.0]
    extreme_rbm = wrap_rbm(kind, base_rbm())
    unit_rbm = wrap_rbm(kind, base_rbm())
    extreme_vm = falses(2, 1)
    unit_vm = copy(extreme_vm)
    extreme_log = callback_log()
    unit_log = callback_log()

    seed!(seed)
    train_pcd!(
        kind, extreme_rbm, data, extreme_wts, extreme_vm,
        CountingDescent(0.01, Ref(0)), extreme_log.callback;
        iters = 2, batchsize = 1,
    )
    seed!(seed)
    train_pcd!(
        kind, unit_rbm, data, unit_wts, unit_vm,
        CountingDescent(0.01, Ref(0)), unit_log.callback;
        iters = 2, batchsize = 1,
    )

    @test model_state(extreme_rbm) == model_state(unit_rbm)
    @test extreme_vm == unit_vm
    # weights are normalized once by their maximum at the training boundary,
    # so callbacks observe the prepared (normalized) weights
    @test extreme_log.weights == unit_log.weights == [[1.0], [1.0]]
    @test all_finite(extreme_rbm)
end

@testset "wmean is a plain weighted mean" begin
    # Weight hygiene (validation, zero-weight filtering, overflow
    # normalization) happens once per training run in
    # `_prepare_training_data`, not per `wmean` call. `wmean` itself is a
    # plain matmul-shaped weighted mean.
    @test RBMs.wmean([1.0, 3.0]; wts = [1.0, 3.0]) ≈ 2.5
    @test RBMs.wmean([1.0, 3.0]; wts = Any[1.0, 3.0]) ≈ 2.5
    @test RBMs.wmean([1.0, 3.0]; wts = fill(big"1e400", 2)) ≈ 2.0
    # zero weights annihilate finite samples exactly...
    @test RBMs.wmean([1.0, 2.0, 4.0]; wts = [0.0, 1.0, 3.0]) ≈ 3.5
    # ...but no longer mask non-finite samples (the training boundary drops
    # zero-weight samples before any kernel sees them)
    @test isnan(RBMs.wmean([NaN, 2.0]; wts = [0.0, 1.0]))

    # the lazy uniform default reduces like a plain mean, without promoting
    A = rand(Float32, 2, 5)
    @test RBMs.wmean(A; dims = 2) ≈ mean(A; dims = 2)
    @test RBMs.wmean(A; dims = 2) isa Matrix{Float32}
    @test RBMs.wmean(A) ≈ mean(A)
    @test RBMs.wmean(A) isa Float32
end

@testset "training weights are prepared once per run" begin
    data = zeros(2, 3)

    # lazy uniform weights skip filtering and normalization entirely
    prepared_data, prepared_wts, wts_mean, batchsize =
        RBMs._prepare_training_data(data, Ones{Bool}(3); batchsize = 5)
    @test prepared_data === data
    @test prepared_wts isa Ones
    @test wts_mean == 1.0
    @test batchsize == 3 # clamped to the number of samples

    # real weights are normalized by their maximum, preserving their float eltype
    for W in (Float32, Float64)
        wts = W[1, 3]
        _, prepared_wts, wts_mean, _ =
            RBMs._prepare_training_data(zeros(W, 1, 2), wts; batchsize = 1)
        @test prepared_wts == W[1, 3] ./ 3
        @test eltype(prepared_wts) == W
        @test wts_mean ≈ (Float64(prepared_wts[1]) + 1) / 2
        batch_weight = RBMs._batch_weight(prepared_wts[1:1], wts_mean)
        @test batch_weight ≈ Float64(prepared_wts[1]) / wts_mean
    end

    # extreme finite weights cannot overflow the training-loop reductions
    for extreme in (floatmax(Float64), typemax(UInt128))
        wts = [zero(extreme), extreme, extreme]
        _, prepared_wts, wts_mean, _ =
            RBMs._prepare_training_data(zeros(1, 3), wts; batchsize = 1)
        @test prepared_wts == [1.0, 1.0]
        @test wts_mean == 1.0
    end

    # narrow weight eltypes are widened to at least Float32, so Float16
    # weights can neither overflow their own sums nor lose moderate ratios
    _, prepared_wts, wts_mean, _ = RBMs._prepare_training_data(
        zeros(Float16, 1, 70_000), ones(Float16, 70_000); batchsize = 1
    )
    @test eltype(prepared_wts) == Float32
    @test wts_mean == 1.0
    tiny16 = nextfloat(zero(Float16))
    _, prepared_wts, _, _ = RBMs._prepare_training_data(
        Float16[1 2], Float16[tiny16, floatmax(Float16)]; batchsize = 1
    )
    @test eltype(prepared_wts) == Float32
    @test length(prepared_wts) == 2
    @test prepared_wts[1] > 0 # the moderate ratio survives in Float32

    # positive weights whose ratio to the maximum underflows even the widened
    # eltype are dropped like zero weights, so no minibatch can end up with
    # weights summing to zero
    tiny32 = nextfloat(zero(Float32))
    prepared_data, prepared_wts, _, _ = RBMs._prepare_training_data(
        Float32[1 2], Float32[tiny32, floatmax(Float32)]; batchsize = 1
    )
    @test prepared_wts == [1.0f0]
    @test prepared_data == Float32[2;;]

    # weights wider than Float64 keep their representable ratios in the
    # minibatch bias correction
    big_wts = BigFloat[big"1e-400", big"1.0"]
    _, prepared_wts, wts_mean, _ = RBMs._prepare_training_data(
        zeros(1, 2), big_wts; batchsize = 1
    )
    @test wts_mean isa BigFloat
    batch_weight = RBMs._batch_weight(prepared_wts[1:1], wts_mean)
    @test batch_weight > 0
    @test batch_weight ≈ big"2e-400" rtol = 1.0e-6

    # lazy uniform weights must still be real-valued to skip validation
    @test_throws ArgumentError RBMs._prepare_training_data(
        zeros(1, 2), Ones{ComplexF64}(2); batchsize = 1
    )
end

@testset "explicitly passed moments are used as given" begin
    rbm = base_rbm()
    data = [0.0 1.0; 1.0 0.0]
    moments = RBMs.moments_from_samples(rbm.visible, data)
    pcd!(rbm, data; batchsize = 2, iters = 1, moments, zerosum = false, rescale = false)
    @test all_finite(rbm)
end

@testset "minibatch bias correction preserves representable products" begin
    ∂ = RBMs.∂RBM(
        fill(Float16(1.0e-3), 1, 2), fill(Float16(1.0e-3), 1, 1), fill(Float16(1.0e-3), 2, 1)
    )
    # a correction beyond floatmax(Float16) with a representable product
    scaled = RBMs._scale_gradient(∂, 1.0e5, Float16)
    @test eltype(scaled.w) == Float16
    @test scaled.w ≈ fill(Float16(100), 2, 1) rtol = 0.01
    # the common representable case converts the scalar once
    scaled = RBMs._scale_gradient(∂, 2.0, Float16)
    @test eltype(scaled.w) == Float16
    @test scaled.w ≈ fill(Float16(2.0e-3), 2, 1) rtol = 0.01
end

@testset "initialize! weight hygiene" begin
    data = [1.0 0.0 1.0; 0.0 1.0 1.0]

    # explicit `wts = nothing` is still accepted, for the RBM and layer methods
    rbm = base_rbm()
    initialize!(rbm, data; wts = nothing)
    @test all_finite(rbm)
    layer = RBMs.Binary((2,))
    initialize!(layer, data; wts = nothing)
    @test all(isfinite, layer.par)

    # extreme finite weights are normalized at the boundary and cannot
    # overflow the moment-matching reductions
    extreme_rbm = base_rbm()
    unit_rbm = base_rbm()
    seed!(303)
    initialize!(extreme_rbm, data; wts = fill(floatmax(Float64), 3))
    seed!(303)
    initialize!(unit_rbm, data; wts = ones(3))
    @test all_finite(extreme_rbm)
    @test model_state(extreme_rbm) == model_state(unit_rbm)

    # zero-weight samples are dropped at the boundary, so non-finite data
    # attached to them cannot poison the moment-matching reductions
    nan_layer = Binary((1,))
    initialize!(nan_layer, [NaN 1.0]; wts = [0.0, 1.0])
    @test all(isfinite, nan_layer.par)

    # multi-dimensional batches are flattened and filtered the same way
    grid_layer = Binary((1,))
    initialize!(grid_layer, reshape([NaN, 1.0, 1.0, 1.0], 1, 2, 2); wts = [0.0 1.0; 1.0 1.0])
    @test all(isfinite, grid_layer.par)
    nan_rbm = base_rbm()
    initialize!(nan_rbm, [NaN 1.0; NaN 0.0]; wts = [0.0, 1.0])
    @test all_finite(nan_rbm)

    # invalid weights fail loudly
    @test_throws ArgumentError initialize!(base_rbm(), data; wts = [1.0, -1.0, 1.0])
    @test_throws ArgumentError initialize!(base_rbm(), data; wts = [1.0, NaN, 1.0])
    # undersized weights are rejected instead of silently truncating the data
    @test_throws DimensionMismatch initialize!(base_rbm(), data; wts = [0.0, 1.0])
end

@testset "∂free_energy with zero weights on finite samples" begin
    # zero weights contribute exactly zero for finite data; non-finite data on
    # zero-weight samples is handled by the training boundary, which drops
    # those samples before the gradient kernels run
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

function train_sparse_pcd!(::Val{:plain}, rbm, data, wts, vm, optim, callback)
    return pcd!(
        rbm, data;
        wts, vm, optim, callback,
        batchsize = 4, iters = 1, steps = 0,
        shuffle = false, zerosum = false, rescale = false,
    )
end

function train_sparse_pcd!(::Val{:centered}, rbm, data, wts, vm, optim, callback)
    return pcd!(
        rbm, data;
        wts, vm, optim, callback,
        batchsize = 4, iters = 1, steps = 0,
        hidden_offset_damping = 0, zerosum = false, rescale = false,
    )
end

function train_sparse_pcd!(::Val{:standardized}, rbm, data, wts, vm, optim, callback)
    return pcd!(
        rbm, data;
        wts, vm, optim, callback,
        batchsize = 4, iters = 1, steps = 0,
        shuffle = false, damping = 0, ϵv = 1, ϵh = 1,
        zerosum = false, rescale_hidden = false,
    )
end

function train_sparse_default_pcd!(::Val{:plain}, rbm, data, wts, optim, callback)
    return pcd!(
        rbm, data;
        wts, optim, callback,
        batchsize = 4, iters = 1, steps = 0,
        shuffle = false, zerosum = false, rescale = false,
    )
end

function train_sparse_default_pcd!(::Val{:centered}, rbm, data, wts, optim, callback)
    return pcd!(
        rbm, data;
        wts, optim, callback,
        batchsize = 4, iters = 1, steps = 0,
        hidden_offset_damping = 0, zerosum = false, rescale = false,
    )
end

function train_sparse_default_pcd!(
        ::Val{:standardized}, rbm, data, wts, optim, callback,
    )
    return pcd!(
        rbm, data;
        wts, optim, callback,
        batchsize = 4, iters = 1, steps = 0,
        shuffle = false, damping = 0, ϵv = 1, ϵh = 1,
        zerosum = false, rescale_hidden = false,
    )
end

@testset "fewer positive samples than batchsize complete ($name PCD)" for (name, kind) in [
        ("plain", Val(:plain)),
        ("centered", Val(:centered)),
        ("standardized", Val(:standardized)),
    ]
    data = [
        NaN NaN NaN NaN 1.0
        NaN NaN NaN NaN 0.0
    ]
    wts = [0.0, 0.0, 0.0, 0.0, 1.0]
    rbm = wrap_rbm(kind, base_rbm())
    vm = falses(2, 4)
    calls = Ref(0)
    log = callback_log()

    train_sparse_pcd!(
        kind, rbm, data, wts, vm, CountingDescent(0.01, calls), log.callback,
    )

    @test log.iterations == [1]
    @test length(only(log.weights)) == 1
    @test all(w -> w > 0, only(log.weights))
    @test calls[] == 3
    @test all_finite(rbm)

    default_rbm = wrap_rbm(kind, base_rbm())
    default_calls = Ref(0)
    fantasy_sizes = Int[]
    seed!(107)
    train_sparse_default_pcd!(
        kind,
        default_rbm,
        data,
        wts,
        CountingDescent(0.01, default_calls),
        (; vm, kwargs...) -> push!(fantasy_sizes, size(vm, ndims(vm))),
    )
    # the default number of fantasy chains is the requested batchsize
    @test fantasy_sizes == [4]
    @test default_calls[] == 3
end
