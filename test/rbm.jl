using Test, Random, Statistics, LinearAlgebra
using Zygote, Flux, FiniteDifferences
using RestrictedBoltzmannMachines, OneHot
using RestrictedBoltzmannMachines: sumdrop, batchsize, batchdims, meandrop,
    init_weights!, Data
using Zygote: @adjoint

n = (5,2)
m = (4,3)
B = (3,1)

rbm = RBM(Binary(n...), Binary(m...))
init!(rbm)
randn!(rbm.vis.θ)
randn!(rbm.hid.θ)

@test size(rbm.vis) == n
@test size(rbm.hid) == m
@test ndims(rbm.vis) == length(n)
@test ndims(rbm.hid) == length(m)
@inferred ndims(rbm.vis)
@inferred ndims(rbm.hid)
@test vdims(rbm) == Tuple(1:ndims(rbm.vis))
@test hdims(rbm) == Tuple(1:ndims(rbm.hid)) .+ ndims(rbm.vis)
@inferred vdims(rbm)
@inferred hdims(rbm)

v = rand((0.0, 1.0), n..., B...)
h = rand((0.0, 1.0), m..., B...)
@test batchdims(rbm.vis, v) == Tuple(ndims(rbm.vis) + 1 : ndims(v))
@test batchdims(rbm.hid, h) == Tuple(ndims(rbm.hid) + 1 : ndims(h))
@inferred batchdims(rbm.vis, v)
@inferred batchdims(rbm.hid, h)
@test batchsize(rbm.vis, v) == B
@test batchsize(rbm.hid, h) == B
@inferred batchsize(rbm.vis, v)
@inferred batchsize(rbm.hid, h)
@test size(energy(rbm.vis, v)) == B
@test size(energy(rbm.hid, h)) == B
@test size(energy(rbm, v, h)) == B

Ev = -sumdrop(rbm.vis.θ .* v; dims=(1,2))
Eh = -sumdrop(rbm.hid.θ .* h; dims=(1,2))
Ew = -[sum(v[i,j,b,c] * rbm.weights[i,j,μ,ν] * h[μ,ν,b,c]
           for i=1:n[1], j=1:n[2], μ=1:m[1], ν=1:m[2])
       for b=1:B[1], c=1:B[2]]
@test energy(rbm.vis, v) ≈ Ev
@test energy(rbm.hid, h) ≈ Eh
@test energy(rbm, v, h) ≈ Ev + Eh + Ew

v1 = sample_v_from_v(rbm, v)
@test size(v1) == size(v)
h1 = sample_h_from_h(rbm, h)
@test size(h1) == size(h)

@test size(free_energy(rbm, v)) == B
@test isfinite(reconstruction_error(rbm, v))

@inferred free_energy(rbm, v)
@inferred sample_v_from_v(rbm, v)
@inferred sample_h_from_h(rbm, h)
@inferred sample_v_from_v(rbm, v)
@inferred sample_h_from_h(rbm, h)

@testset "init!" begin
    x = rand((0.0, 1.0), 2,2, 100)
    layer = Binary(2,2)
    init!(layer, x; eps=0)
    @test transfer_mean(layer) ≈ meandrop(x; dims=ndims(x))

    x = rand((-1.0, 1.0), 2,2, 100)
    layer = Spin(2,2)
    init!(layer, x; eps=0)
    @test transfer_mean(layer) ≈ meandrop(x; dims=ndims(x))
end

@testset "single batch" begin
    n = (5,2)
    m = (4,3)
    rbm = RBM(Binary(n...), Binary(m...))
    init_weights!(rbm)
    randn!(rbm.vis.θ)
    v = random(rbm.vis)
    @test size(v) == size(rbm.vis)
    @test free_energy(rbm, v) isa Number
end

@testset "inputs_v_to_h gradient" begin
    # with batch dimensions
    w = randn(2,4,5,3)
    v = rand(Bool,2,4, 2,1)
    testfun(w) = sum(sin.(inputs_v_to_h(RBM(Binary(2,4), Binary(5,3), w), v)))
    (∂w,) = gradient(testfun, w)
    p = randn(size(w))
    @test central_fdm(5,1)(ϵ -> testfun(w + ϵ * p), 0) ≈ sum(∂w .* p)

    # without batch dimensions
    w = randn(2,4,5,3)
    v = rand(Bool,2,4)
    testfun(w) = sum(sin.(inputs_v_to_h(RBM(Binary(2,4), Binary(5,3), w), v)))
    (∂w,) = gradient(testfun, w)
    p = randn(size(w))
    @test central_fdm(5,1)(ϵ -> testfun(w + ϵ * p), 0) ≈ sum(∂w .* p)
end

@testset "inputs_h_to_v gradient" begin
    # with batch dimensions
    w = randn(2,4,5,3)
    p = randn(size(w))
    h = rand(Bool,5,3, 2,1)
    hsel = CartesianIndices((1:2, 1:3))
    testfun(w) = sum(sin.(inputs_h_to_v(RBM(Binary(2,4), Binary(5,3), w), h)))
    (∂w,) = gradient(testfun, w)
    @test central_fdm(5,1)(ϵ -> testfun(w + ϵ * p), 0) ≈ sum(∂w .* p)
    testfun(w) = sum(sin.(inputs_h_to_v(RBM(Binary(2,4), Binary(5,3), w), h, hsel)))
    (∂w,) = gradient(testfun, w)
    @test central_fdm(5,1)(ϵ -> testfun(w + ϵ * p), 0) ≈ sum(∂w .* p)

    # without batch dimensions
    h = rand(Bool,5,3)
    testfun(w) = sum(sin.(inputs_h_to_v(RBM(Binary(2,4), Binary(5,3), w), h)))
    (∂w,) = gradient(testfun, w)
    @test central_fdm(5,1)(ϵ -> testfun(w + ϵ * p), 0) ≈ sum(∂w .* p)
    testfun(w) = sum(sin.(inputs_h_to_v(RBM(Binary(2,4), Binary(5,3), w), h, hsel)))
    (∂w,) = gradient(testfun, w)
    @test central_fdm(5,1)(ϵ -> testfun(w + ϵ * p), 0) ≈ sum(∂w .* p)
end

@testset "training" begin
    train_x = rand((0.0, 1.0), 2,2, 100)
    train_w = rand(100)
    data = Data((v = train_x, w = train_w); batchsize=3)
    rbm = RBM(Binary(2,2), Gaussian(10))
    init!(rbm, train_x; eps = 0)
    @test transfer_mean(rbm.vis) ≈ meandrop(train_x; dims=ndims(train_x))
    @time train!(rbm, data; iters=500 * 3)
end

@testset "cd training, simple teacher / student" begin
    Random.seed!(3)
    N = 10; B = 1000
    teacher = RBM(Binary(N), Gaussian(1))
    randn!(teacher.weights)
    gauge!(teacher)
    @test norm(teacher.weights) ≈ 1
    train_x = rand((0., 1.), N, B)
    train_x = sample_v_from_v(teacher, train_x; steps=1000)
    train_data = Data((v = train_x, w = ones(B)); batchsize=32)

    student = RBM(Binary(N), Gaussian(1))
    randn!(student.weights)
    gauge!(student)
    @test norm(teacher.weights) ≈ 1
    ps = params(student.weights)
    train!(student, train_data, PCD(5); iters=10000 * 32, ps = ps, opt = Flux.ADAM())
    @test norm(teacher.weights) ≈ 1
    @show dot(teacher.weights, student.weights)
    @test abs(dot(teacher.weights, student.weights)) ≥ 0.8
end

@testset "cd training, teacher / student, with Potts" begin
    Random.seed!(3)
    q = 3; N = 10; B = 1000
    teacher = RBM(Potts(q,N), Gaussian(1))
    randn!(teacher.weights)
    gauge!(teacher)
    @test norm(teacher.weights) ≈ 1
    train_x = float(OneHot.encode(rand(1:q, N, B), q))
    train_x = sample_v_from_v(teacher, train_x; steps=1000)
    train_data = Data((v = train_x, w = ones(B)); batchsize=32)

    student = RBM(Potts(q,N), Gaussian(1))
    randn!(student.weights)
    gauge!(student)
    @test norm(teacher.weights) ≈ 1
    ps = params(student.weights)
    train!(student, train_data, PCD(5); iters=10000 * 32, ps = ps, opt = Flux.ADAM())
    @test norm(teacher.weights) ≈ 1
    @show dot(teacher.weights, student.weights)
    @test abs(dot(teacher.weights, student.weights)) ≥ 0.8
end

@testset "cd loss gradient, Potts/Gaussian" begin
    Random.seed!(3)
    q = 3; N = 10; M = 5; B = 100
    vd = float(OneHot.encode(rand(1:q, N, B)))
    vm = float(OneHot.encode(rand(1:q, N, B)))
    wd = rand(B)
    rbm = RBM(Potts(q, N), Gaussian(M))
    randn!(rbm.weights)
    randn!(rbm.vis.θ)
    randn!(rbm.hid.θ)
    rand!(rbm.hid.γ)
    gauge!(rbm)

    ps = params(rbm)
    @test rbm.weights ∈ ps
    @test rbm.vis.θ ∈ ps
    @test rbm.hid.θ ∈ ps
    @test rbm.hid.γ ∈ ps
    gs = gradient(ps) do
        contrastive_divergence(rbm, vd, vm, wd)
    end

    pg = randn(q,N); pθ = randn(M); pγ = rand(M); pw = randn(q,N,M);
    Δ = central_fdm(5,1)(0) do ϵ
        rbm_ = RBM(Potts(rbm.vis.θ .+ ϵ .* pg),
                  Gaussian(rbm.hid.θ .+ ϵ .* pθ, rbm.hid.γ .+ ϵ .* pγ),
                  rbm.weights .+ ϵ .* pw)
        contrastive_divergence(rbm_, vd, vm, wd)
    end
    @test Δ ≈ dot(gs[rbm.weights], pw) + dot(gs[rbm.vis.θ], pg) +
              dot(gs[rbm.hid.θ], pθ) + dot(gs[rbm.hid.γ], pγ)
end

@testset "cd loss gradient, Spin/Gaussian" begin
    Random.seed!(3)
    N = (3,10); M = (5,2); B = 100
    vd = rand((-1.,1.), N..., B)
    vm = rand((-1.,1.), N..., B)
    wd = rand(B)
    rbm = RBM(Spin(N...), Gaussian(M...))
    randn!(rbm.weights)
    randn!(rbm.vis.θ)
    randn!(rbm.hid.θ)
    rand!(rbm.hid.γ)
    gauge!(rbm)

    ps = params(rbm)
    @test rbm.weights ∈ ps
    @test rbm.vis.θ ∈ ps
    @test rbm.hid.θ ∈ ps
    @test rbm.hid.γ ∈ ps
    gs = gradient(ps) do
        contrastive_divergence(rbm, vd, vm, wd)
    end

    pg = randn(N...); pθ = randn(M...); pγ = rand(M...); pw = randn(N...,M...);
    Δ = central_fdm(5,1)(0) do ϵ
        rbm_ = RBM(Spin(rbm.vis.θ .+ ϵ .* pg),
                   Gaussian(rbm.hid.θ .+ ϵ .* pθ, rbm.hid.γ .+ ϵ .* pγ),
                   rbm.weights .+ ϵ .* pw)
        contrastive_divergence(rbm_, vd, vm, wd)
    end
    @test Δ ≈ dot(gs[rbm.weights], pw) + dot(gs[rbm.vis.θ], pg) +
              dot(gs[rbm.hid.θ], pθ) + dot(gs[rbm.hid.γ], pγ)
end

@testset "cd loss gradient, Potts/dReLU" begin
    Random.seed!(3)
    q = 3; N = 10; M = 5; B = 100
    vd = float(OneHot.encode(rand(1:q, N, B)))
    vm = float(OneHot.encode(rand(1:q, N, B)))
    wd = rand(B)
    rbm = RBM(Potts(q, N), dReLU(M))
    randn!(rbm.weights)
    randn!(rbm.vis.θ)
    randn!(rbm.hid.θp)
    randn!(rbm.hid.θn)
    rand!(rbm.hid.γp)
    rand!(rbm.hid.γn)
    gauge!(rbm)

    ps = params(rbm)
    @test rbm.weights ∈ ps
    @test rbm.vis.θ ∈ ps
    @test rbm.hid.θp ∈ ps
    @test rbm.hid.θn ∈ ps
    @test rbm.hid.γp ∈ ps
    @test rbm.hid.γn ∈ ps
    gs = gradient(ps) do
        contrastive_divergence(rbm, vd, vm, wd)
    end

    pw = randn(q,N,M); pg = randn(q,N);
    pθp = randn(M); pγp = randn(M);
    pθn = randn(M); pγn = randn(M);
    Δ = central_fdm(5,1)(0) do ϵ
        rbm_ = RBM(Potts(rbm.vis.θ .+ ϵ .* pg),
                   dReLU(rbm.hid.θp .+ ϵ .* pθp, rbm.hid.θn .+ ϵ .* pθn,
                         rbm.hid.γp .+ ϵ .* pγp, rbm.hid.γn .+ ϵ .* pγn),
                   rbm.weights .+ ϵ .* pw)
        contrastive_divergence(rbm_, vd, vm, wd)
    end
    @test Δ ≈ dot(gs[rbm.weights], pw) + dot(gs[rbm.vis.θ], pg) +
              dot(gs[rbm.hid.θp], pθp) + dot(gs[rbm.hid.θn], pθn) +
              dot(gs[rbm.hid.γp], pγp) + dot(gs[rbm.hid.γn], pγn)
end
