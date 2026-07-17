using Optimisers: Descent
using RestrictedBoltzmannMachines: Binary, RBM, center, cgfs, grad2var, pReLU,
    pcd!, standardize, xReLU, ∂energy_from_moments
using Test: @test, @testset

function check_prelu_error(f; context::AbstractString)
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
        @test occursin(context, msg)
    end
    return nothing
end

function unit_prelu(η)
    return pReLU(; θ = [0.2], γ = [1.3], Δ = [0.4], η = [η])
end

function boundary_rbm()
    return RBM(Binary((1,)), unit_prelu(0.99), ones(1, 1))
end

pcd_model(::Val{:plain}) = boundary_rbm()
pcd_model(::Val{:centered}) = center(boundary_rbm())
pcd_model(::Val{:standardized}) = standardize(boundary_rbm())

function run_pcd!(
    ::Val{:plain}, rbm, data, vm; iters::Int, callback,
)
    return pcd!(
        rbm, data;
        batchsize = 1, iters, steps = 0, vm, callback,
        optim = Descent(1e-3), rescale = false, zerosum = false, shuffle = false,
    )
end

function run_pcd!(
    ::Val{:centered}, rbm, data, vm; iters::Int, callback,
)
    return pcd!(
        rbm, data;
        batchsize = 1, iters, steps = 0, vm, callback,
        optim = Descent(1e-3), rescale = false, zerosum = false,
        hidden_offset_damping = 0,
    )
end

function run_pcd!(
    ::Val{:standardized}, rbm, data, vm; iters::Int, callback,
)
    return pcd!(
        rbm, data;
        batchsize = 1, iters, steps = 0, vm, callback,
        optim = Descent(1e-3), rescale_hidden = false, zerosum = false,
        shuffle = false, damping = 0, ϵv = 1,
    )
end

@testset "pReLU η construction and evaluation" begin
    for η in (-Inf, -1.1, -1.0, 1.0, 1.1, Inf, NaN)
        check_prelu_error(() -> unit_prelu(η); context = "construction")
    end

    integer_par = zeros(Int, 4, 1)
    integer_par[2, 1] = 1
    integer_par[4, 1] = typemin(Int)
    check_prelu_error(() -> pReLU(integer_par); context = "construction")

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
    check_prelu_error(() -> cgfs(invalid); context = "evaluation or conversion")
    check_prelu_error(
        () -> ∂energy_from_moments(invalid, zeros(4, 1));
        context = "evaluation or conversion",
    )
    check_prelu_error(() -> xReLU(invalid); context = "evaluation or conversion")
    check_prelu_error(
        () -> grad2var(invalid, zeros(4, 1));
        context = "evaluation or conversion",
    )
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
        () -> run_pcd!(case, initially_invalid, data, copy(vm); iters = 0, callback);
        context = "before pcd! training",
    )
    @test !callback_called[]

    crossing = pcd_model(case)
    check_prelu_error(
        () -> run_pcd!(case, crossing, data, copy(vm); iters = 1, callback);
        context = "after a pcd! optimizer update",
    )
    @test only(crossing.hidden.η) > 1
    @test !callback_called[]
end

@testset "pReLU η validation checks the visible layer" begin
    rbm = RBM(unit_prelu(0.0), Binary((1,)), ones(1, 1))
    rbm.visible.η .= Inf
    check_prelu_error(
        () -> run_pcd!(
            Val(:plain), rbm, zeros(1, 1), zeros(1, 1);
            iters = 0, callback = (; _...) -> nothing,
        );
        context = "visible layer before pcd! training",
    )
end
