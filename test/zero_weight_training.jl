using Test: @test, @testset, @test_throws
using Random: seed!
import Optimisers
import RestrictedBoltzmannMachines as RBMs
using RestrictedBoltzmannMachines: RBM, Binary, Gaussian, BinaryRBM,
    CenteredRBM, StandardizedRBM, center, standardize, pcd!

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
    for (bad_wts, err) in (
            ([2.0, -1.0], ArgumentError),
            ([1.0, NaN], ArgumentError),
            ([1.0, Inf], ArgumentError),
            (ComplexF64[1, 1], MethodError), # complex weights have no ordering
        )
        @test_throws ArgumentError RBMs._prepare_training_data(data, bad_wts; batchsize = 1)
        rbm = base_rbm()
        before = model_state(rbm)
        @test_throws err pcd!(
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
    @test extreme_log.weights == [[extreme_weight], [extreme_weight]]
    @test unit_log.weights == [[1.0], [1.0]]
    @test all_finite(extreme_rbm)
end

@testset "wmean ignores zero weights and extreme scales" begin
    # zero weights annihilate their samples exactly, even non-finite ones
    @test RBMs.wmean([NaN, 2.0, 4.0]; wts = [0.0, 1.0, 3.0]) ≈ 3.5
    # extreme finite weights are normalized internally and cannot overflow
    @test RBMs.wmean([1.0, 3.0]; wts = fill(floatmax(Float64), 2)) ≈ 2.0
    @test RBMs.wmean([1.0, 3.0]; wts = fill(typemax(UInt128), 2)) ≈ 2.0
    # finite weights wider than Float64 keep their wide accumulator,
    # even behind an abstract eltype
    @test RBMs.wmean([1.0, 3.0]; wts = fill(big"1e400", 2)) ≈ 2.0
    @test RBMs.wmean([1.0, 3.0]; wts = Real[big"1e400", big"1e400"]) ≈ 2.0
    @test RBMs.wmean([1.0, 3.0]; wts = Any[1.0, 3.0]) ≈ 2.5
    # Float16 weights are accumulated in a wider type (naive Float16 sums overflow)
    n = 70_000
    A = reshape(repeat(Float16[1, 3], n ÷ 2), 1, n)
    @test RBMs.wmean(A; wts = ones(Float16, n), dims = 2) ≈ [2]
end

@testset "∂free_energy ignores zero-weight samples exactly" begin
    rbm = base_rbm()
    v = [
        NaN 0.0 1.0
        NaN 1.0 0.0
    ]
    wts = [0.0, 1.0, 2.0]
    ∂ = RBMs.∂free_energy(rbm, v; wts)
    ∂ref = RBMs.∂free_energy(rbm, v[:, 2:3]; wts = wts[2:3])
    @test ∂.visible ≈ ∂ref.visible
    @test ∂.hidden ≈ ∂ref.hidden
    @test ∂.w ≈ ∂ref.w
end

@testset "narrow mixed-range weights keep a positive batch weight ($W)" for W in (Float16, Float32)
    small = nextfloat(zero(W))
    large = floatmax(W)
    wts = W[small, large]
    data = zeros(W, 1, length(wts))

    _, raw_wts, normalization, _ = RBMs._prepare_training_data(data, wts; batchsize = 1)
    @test raw_wts === wts

    ratio = Float64(small) / Float64(large)
    batch_weight = RBMs._batch_weight(raw_wts[1:1], normalization)
    @test batch_weight > 0
    @test batch_weight ≈ 2ratio / (1 + ratio)
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

