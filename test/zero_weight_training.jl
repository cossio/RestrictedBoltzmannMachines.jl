using Test: @test, @testset, @test_throws
using Random: seed!
import Optimisers
import RestrictedBoltzmannMachines as RBMs
using RestrictedBoltzmannMachines: RBM, Binary, Gaussian, BinaryRBM,
    CenteredRBM, StandardizedRBM, center, standardize, pcd!, ucd!

struct CountingDescent{T,R} <: Optimisers.AbstractRule
    eta::T
    calls::R
end

Optimisers.init(::CountingDescent, x::AbstractArray) = nothing

function Optimisers.apply!(rule::CountingDescent, state, x, dx)
    rule.calls[] += 1
    return state, rule.eta .* dx
end

struct MutatingInitRule{R} <: Optimisers.AbstractRule
    calls::R
end

function Optimisers.init(rule::MutatingInitRule, x::AbstractArray)
    rule.calls[] += 1
    x .+= one(eltype(x))
    return nothing
end

Optimisers.apply!(::MutatingInitRule, state, x, dx) = (state, dx)

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
        steps = 1, hidden_offset_damping = 1//4,
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
        steps = 1, shuffle = false, damping = 1//4, ϵv = 0.1, ϵh = 0.1,
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

@testset "zero-weight UCD matches filtering" begin
    data, wts = weighted_data()
    positive = findall(!iszero, wts)
    filtered_data = data[:, positive]
    filtered_wts = wts[positive]

    mixed_rbm = base_rbm()
    filtered_rbm = deepcopy(mixed_rbm)
    mixed_calls = Ref(0)
    filtered_calls = Ref(0)
    mixed_log = callback_log()
    filtered_log = callback_log()
    mixed_chain_stats = Tuple{Float64,Float64}[]
    filtered_chain_stats = Tuple{Float64,Float64}[]
    mixed_callback = (; meeting_steps, discarded, kwargs...) -> begin
        mixed_log.callback(; kwargs...)
        push!(mixed_chain_stats, (meeting_steps, discarded))
        return nothing
    end
    filtered_callback = (; meeting_steps, discarded, kwargs...) -> begin
        filtered_log.callback(; kwargs...)
        push!(filtered_chain_stats, (meeting_steps, discarded))
        return nothing
    end

    seed!(104)
    ucd!(
        mixed_rbm, data;
        wts, batchsize = 2, iters = 2, nchains = 1,
        min_steps = 1, max_steps = 64, max_resamples = 100,
        optim = CountingDescent(0.01, mixed_calls),
        callback = mixed_callback, shuffle = false, zerosum = false, rescale = false,
    )
    seed!(104)
    ucd!(
        filtered_rbm, filtered_data;
        wts = filtered_wts, batchsize = 2, iters = 2, nchains = 1,
        min_steps = 1, max_steps = 64, max_resamples = 100,
        optim = CountingDescent(0.01, filtered_calls),
        callback = filtered_callback, shuffle = false, zerosum = false, rescale = false,
    )

    @test mixed_log.iterations == filtered_log.iterations == 1:2
    @test all(wd -> all(w -> w > 0, wd), mixed_log.weights)
    @test mixed_calls[] == filtered_calls[] == 3 * 2
    @test mixed_chain_stats == filtered_chain_stats
    @test model_state(mixed_rbm) == model_state(filtered_rbm)
    @test all_finite(mixed_rbm)
end

@testset "invalid training weights" begin
    data = zeros(2, 2)
    for bad_wts in (
        [2.0, -1.0],
        [1.0, NaN],
        [1.0, Inf],
        ComplexF64[1, 1],
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

@testset "finite extreme weights are scale-stable ($name PCD)" for (name, kind, seed) in [
    ("plain", Val(:plain), 109),
    ("centered", Val(:centered), 110),
    ("standardized", Val(:standardized), 111),
]
    data = [
        NaN 0.0 1.0
        NaN 1.0 0.0
    ]
    extreme_wts = [0.0, floatmax(Float64), floatmax(Float64)]
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
    @test extreme_log.weights == [[floatmax(Float64)], [floatmax(Float64)]]
    @test unit_log.weights == [[1.0], [1.0]]
    @test all_finite(extreme_rbm)
end

@testset "finite extreme weights are scale-stable (UCD)" begin
    data = [
        NaN 0.0 1.0
        NaN 1.0 0.0
    ]
    extreme_wts = [0.0, floatmax(Float64), floatmax(Float64)]
    unit_wts = [0.0, 1.0, 1.0]
    extreme_rbm = base_rbm()
    unit_rbm = base_rbm()
    extreme_log = callback_log()
    unit_log = callback_log()
    common = (;
        batchsize = 1,
        iters = 2,
        nchains = 1,
        min_steps = 1,
        max_steps = 64,
        max_resamples = 100,
        shuffle = false,
        zerosum = false,
        rescale = false,
    )

    seed!(112)
    ucd!(
        extreme_rbm, data;
        common..., wts = extreme_wts,
        optim = CountingDescent(0.01, Ref(0)),
        callback = extreme_log.callback,
    )
    seed!(112)
    ucd!(
        unit_rbm, data;
        common..., wts = unit_wts,
        optim = CountingDescent(0.01, Ref(0)),
        callback = unit_log.callback,
    )

    @test model_state(extreme_rbm) == model_state(unit_rbm)
    @test extreme_log.weights == [[floatmax(Float64)], [floatmax(Float64)]]
    @test unit_log.weights == [[1.0], [1.0]]
    @test all_finite(extreme_rbm)
end

@testset "Float16 weight normalization uses a safe accumulator" begin
    data = zeros(Float16, 1, 70_000)
    wts = ones(Float16, 70_000)
    _, raw_wts, training_wts, normalization, _ =
        RBMs._prepare_training_data(data, wts; batchsize = length(wts))

    @test raw_wts === wts
    @test eltype(training_wts) === Float32
    @test normalization.mean == 1.0f0

    training_batch_wts, batch_weight =
        RBMs._prepare_training_batch(raw_wts, normalization)
    @test eltype(training_batch_wts) === Float32
    @test batch_weight == 1
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
    init_calls = Ref(0)
    callbacks = Ref(0)

    seed!(106)
    expected_next_random = rand()
    seed!(106)
    @test_throws ArgumentError pcd!(
        rbm, data;
        wts, batchsize = 2, iters = 1, steps = 1,
        optim = MutatingInitRule(init_calls),
        callback = (; kwargs...) -> (callbacks[] += 1),
    )
    @test model_state(rbm) == before
    @test rand() == expected_next_random
    @test iszero(init_calls[])
    @test iszero(callbacks[])
    return nothing
end

@testset "all-zero weights fail before mutation ($name PCD)" for (name, kind) in [
    ("plain", Val(:plain)),
    ("centered", Val(:centered)),
    ("standardized", Val(:standardized)),
]
    check_all_zero_pcd(kind)
end

@testset "all-zero weights fail before mutation (UCD)" begin
    data = [0.0 1.0; 1.0 0.0]
    rbm = base_rbm()
    before = model_state(rbm)
    init_calls = Ref(0)
    callbacks = Ref(0)

    @test_throws ArgumentError ucd!(
        rbm, data;
        wts = zeros(2), batchsize = 2, iters = 1, nchains = 1,
        optim = MutatingInitRule(init_calls),
        callback = (; kwargs...) -> (callbacks[] += 1),
    )
    @test model_state(rbm) == before
    @test iszero(init_calls[])
    @test iszero(callbacks[])
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
    @test fantasy_sizes == [1]
    @test default_calls[] == 3
end

@testset "fewer positive samples than batchsize complete (UCD)" begin
    data = [
        NaN NaN NaN NaN 1.0
        NaN NaN NaN NaN 0.0
    ]
    wts = [0.0, 0.0, 0.0, 0.0, 1.0]
    rbm = base_rbm()
    calls = Ref(0)
    log = callback_log()

    seed!(105)
    ucd!(
        rbm, data;
        wts, batchsize = 4, iters = 1, nchains = 1,
        min_steps = 1, max_steps = 64, max_resamples = 100,
        optim = CountingDescent(0.01, calls),
        callback = log.callback, shuffle = false, zerosum = false, rescale = false,
    )

    @test log.iterations == [1]
    @test length(only(log.weights)) == 1
    @test all(w -> w > 0, only(log.weights))
    @test calls[] == 3
    @test all_finite(rbm)

    default_rbm = base_rbm()
    explicit_rbm = deepcopy(default_rbm)
    common = (;
        wts,
        batchsize = 4,
        iters = 1,
        min_steps = 1,
        max_steps = 64,
        max_resamples = 100,
        shuffle = false,
        zerosum = false,
        rescale = false,
    )
    seed!(108)
    ucd!(default_rbm, data; common...)
    seed!(108)
    ucd!(explicit_rbm, data; common..., nchains = 1)
    @test model_state(default_rbm) == model_state(explicit_rbm)
end
