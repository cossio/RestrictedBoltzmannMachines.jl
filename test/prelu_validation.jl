import Optimisers
using Optimisers: Descent
using RestrictedBoltzmannMachines: Binary, RBM, center, cgfs, grad2var, pReLU,
    pcd!, standardize, xReLU, ∂energy_from_moments
using Test: @test, @testset

function check_prelu_error(f)
    err = try
        f()
        nothing
    catch err
        err
    end
    @test err isa ArgumentError
    if err isa ArgumentError
        msg = sprint(showerror, err)
        @test occursin("pReLU.η", msg)
        @test occursin("(-1, 1)", msg)
        @test occursin("xReLU", msg)
        @test occursin("nsReLU", msg)
    end
    return nothing
end

function unit_prelu(η)
    return pReLU(; θ = [0.2], γ = [1.3], Δ = [0.4], η = [η])
end

function boundary_rbm()
    return RBM(Binary((1,)), unit_prelu(0.99), ones(1, 1))
end

struct PreluMutatingInitRule{R} <: Optimisers.AbstractRule
    calls::R
end

function Optimisers.init(rule::PreluMutatingInitRule, x::AbstractArray)
    rule.calls[] += 1
    x .+= one(eltype(x))
    return nothing
end

Optimisers.apply!(::PreluMutatingInitRule, state, x, dx) = (state, dx)

pcd_model(::Val{:plain}) = boundary_rbm()
pcd_model(::Val{:centered}) = center(boundary_rbm())
pcd_model(::Val{:standardized}) = standardize(boundary_rbm())

function run_pcd!(
    ::Val{:plain}, rbm, data, vm;
    iters::Int, callback, wts = nothing, optim = Descent(1e-3),
)
    return pcd!(
        rbm, data;
        batchsize = 1, iters, steps = 0, vm, callback, wts, optim,
        rescale = false, zerosum = false, shuffle = false,
    )
end

function run_pcd!(
    ::Val{:centered}, rbm, data, vm;
    iters::Int, callback, wts = nothing, optim = Descent(1e-3),
)
    return pcd!(
        rbm, data;
        batchsize = 1, iters, steps = 0, vm, callback, wts, optim,
        rescale = false, zerosum = false,
        hidden_offset_damping = 0,
    )
end

function run_pcd!(
    ::Val{:standardized}, rbm, data, vm;
    iters::Int, callback, wts = nothing, optim = Descent(1e-3),
)
    return pcd!(
        rbm, data;
        batchsize = 1, iters, steps = 0, vm, callback, wts, optim,
        rescale_hidden = false, zerosum = false,
        shuffle = false, damping = 0, ϵv = 1,
    )
end

function run_pcd_with_default_vm!(::Val{:plain}, rbm, data; callback)
    return pcd!(
        rbm, data;
        batchsize = 1, iters = 0, steps = 0, callback,
        rescale = false, zerosum = false, shuffle = false,
    )
end

function run_pcd_with_default_vm!(::Val{:centered}, rbm, data; callback)
    return pcd!(
        rbm, data;
        batchsize = 1, iters = 0, steps = 0, callback,
        rescale = false, zerosum = false,
        hidden_offset_damping = 0,
    )
end

function run_pcd_with_default_vm!(::Val{:standardized}, rbm, data; callback)
    return pcd!(
        rbm, data;
        batchsize = 1, iters = 0, steps = 0, callback,
        rescale_hidden = false, zerosum = false,
        shuffle = false, damping = 0, ϵv = 1,
    )
end

@testset "pReLU η construction and evaluation" begin
    for η in (-Inf, -1.1, -1.0, 1.0, 1.1, Inf, NaN)
        check_prelu_error(() -> unit_prelu(η))
    end

    integer_par = zeros(Int, 4, 1)
    integer_par[2, 1] = 1
    integer_par[4, 1] = typemin(Int)
    check_prelu_error(() -> pReLU(integer_par))

    for η in (nextfloat(-1.0), prevfloat(1.0))
        valid = unit_prelu(η)
        @test valid isa pReLU
        @test cgfs(valid) isa AbstractArray
    end

    unsigned_par = zeros(UInt, 4, 1)
    unsigned_par[2, 1] = 1
    @test pReLU(unsigned_par) isa pReLU

    invalid = unit_prelu(0.0)
    invalid.η .= 1
    check_prelu_error(() -> cgfs(invalid))
    check_prelu_error(() -> ∂energy_from_moments(invalid, zeros(4, 1)))
    check_prelu_error(() -> xReLU(invalid))
    check_prelu_error(() -> grad2var(invalid, zeros(4, 1)))
end

@testset "pReLU η validation in PCD: $name" for name in (:plain, :centered, :standardized)
    case = Val(name)
    data = falses(1, 1)
    vm = trues(1, 1)
    callback_called = Ref(false)
    callback = (; _...) -> (callback_called[] = true)

    initially_invalid = pcd_model(case)
    initially_invalid.hidden.η .= NaN
    check_prelu_error(
        () -> run_pcd!(case, initially_invalid, data, copy(vm); iters = 0, callback)
    )
    @test !callback_called[]

    crossing = pcd_model(case)
    check_prelu_error(
        () -> run_pcd!(case, crossing, data, copy(vm); iters = 1, callback)
    )
    @test only(crossing.hidden.η) > 1
    @test !callback_called[]
end

@testset "pReLU η validation with zero-weight filtering: $name" for
    name in (:plain, :centered, :standardized)

    case = Val(name)
    data = reshape([NaN, 0.0], 1, 2)
    vm = trues(1, 1)
    callback_called = Ref(false)
    callback = (; _...) -> (callback_called[] = true)

    crossing = pcd_model(case)
    check_prelu_error(
        () -> run_pcd!(
            case, crossing, data, copy(vm);
            iters = 1, callback, wts = [0.0, 1.0],
        )
    )
    @test only(crossing.hidden.η) > 1
    @test !callback_called[]

    initially_invalid = pcd_model(case)
    initially_invalid.hidden.η .= NaN
    init_calls = Ref(0)
    check_prelu_error(
        () -> run_pcd!(
            case, initially_invalid, data, copy(vm);
            iters = 1, callback, wts = zeros(2),
            optim = PreluMutatingInitRule(init_calls),
        )
    )
    @test iszero(init_calls[])
    @test !callback_called[]
end

@testset "pReLU η validation checks the visible layer before default fantasy initialization: $name" for
    name in (:plain, :centered, :standardized)

    rbm = RBM(unit_prelu(0.0), Binary((1,)), ones(1, 1))
    name === :centered && (rbm = center(rbm))
    name === :standardized && (rbm = standardize(rbm))
    rbm.visible.η .= Inf
    callback_called = Ref(false)
    check_prelu_error(
        () -> run_pcd_with_default_vm!(
            Val(name), rbm, zeros(1, 1);
            callback = (; _...) -> (callback_called[] = true),
        )
    )
    @test !callback_called[]
end
