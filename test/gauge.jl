include("tests_init.jl")

n = (5,2,3,5)
m = (4,3,2)
B = (3,1)

@testset "zerosum" begin
    rbm = RBM(Potts(5,6), Gaussian(3))
    rand!(rbm.visible.θ); rand!(rbm.weights);
    rand!(rbm.hidden.θ); rand!(rbm.hidden.γ);

    rbm.visible.θ .+= 1
    @test norm(sum(rbm.visible.θ; dims=1)) > 1
    zerosum!(rbm.visible)
    @test norm(sum(rbm.visible.θ; dims=1)) < 1e-10

    rbm.weights .+= 1
    @test norm(sum(rbm.weights; dims=1)) > 1
    zerosum!(rbm)
    @test norm(sum(rbm.weights; dims=1)) < 1e-10
end

@testset "cd gauge invariance Binary / Gaussian" begin
    rbm = RBM(Binary(5), Gaussian(3))
    randn!(rbm.visible.θ); randn!(rbm.weights);
    randn!(rbm.hidden.θ); rand!(rbm.hidden.γ);
    vd = RBMs.sample_from_inputs(rbm.visible, zeros(size(rbm.visible)..., 10))
    vm = RBMs.sample_from_inputs(rbm.visible, zeros(size(rbm.visible)..., 10))
    λ, κ = rand(size(rbm.hidden)...), randn(size(rbm.hidden)...)
    dλ, dκ = gradient(λ, κ) do λ, κ
        g_ = rbm.visible.θ .+ tensormul_lf(rbm.weights, κ, Val(ndims(rbm.hidden)))
        w_ = rbm.weights ./ reshape(λ, ones(Int, ndims(rbm.visible))..., size(rbm.hidden)...)
        θ_ = (rbm.hidden.θ .- rbm.hidden.γ .* κ) ./ λ
        γ_ = rbm.hidden.γ ./ λ.^2
        rbm_ = RBM(Binary(g_), Gaussian(θ_, γ_), w_)
        contrastive_divergence(rbm_, vd, vm)
    end
    @test norm(dλ) ≤ 1e-10
    @test norm(dκ) ≤ 1e-10
end

@testset "gauge Binary / ReLU" begin
    rbm = RBM(Binary(n...), Gaussian(m...))
    randn!(rbm.weights)
    randn!(rbm.visible.θ)
    randn!(rbm.hidden.θ)
    rand!(rbm.hidden.γ)

    rbm_ = gauge(rbm)
    @test gauge(rbm_).weights ≈ rbm_.weights
    @test all(sum(rbm_.weights.^2; dims=vdims(rbm)) .≈ 1)

    #= Tests that the machine did not lose expressivity. More precisely,
    compensante rescaling by scaling of hidden units and show that this
    restores original free energies. =#
    rbm_ = rescale(deepcopy(rbm))
    ω = 1 ./ sqrt.(sum(rbm.weights.^2; dims=vdims(rbm)))
    @test rbm_.weights ≈ rbm.weights .* ω
    @test rbm_.vis.θ ≈ rbm.visible.θ
    @test rbm_.hid.θ ≈ rbm.hidden.θ
    @test rbm_.hid.γ ≈ rbm.hidden.γ
    ω = dropdims(ω; dims = vdims(rbm))
    @test size(ω) == size(rbm.hidden)
    rbm_.hid.θ .*= ω
    rbm_.hid.γ .*= ω .* ω
    I = randn(n..., B...)
    v1 = RBMs.sample_from_inputs(rbm.visible, +I)
    v2 = RBMs.sample_from_inputs(rbm.visible, -I)
    Δf = free_energy(rbm,  v1) - free_energy(rbm,  v2)
    Δg = free_energy(rbm_, v1) - free_energy(rbm_, v2)
    @test Δf ≈ Δg
end

@testset "gauge Onehot / ReLU" begin
    rbm = RBM(Potts(n...), ReLU(m...))
    randn!(rbm.weights)
    randn!(rbm.visible.θ)
    randn!(rbm.hidden.θ)
    rand!(rbm.hidden.γ)

    rbm_ = RBMs.gauge(rbm)
    @test RBMs.gauge(rbm_).weights ≈ rbm_.weights
    @test RBMs.gauge(rbm_).vis.θ ≈ rbm_.vis.θ
    @test all(sum(rbm_.weights.^2; dims=vdims(rbm)) .≈ 1)
    @test all(abs.(mean(rbm_.weights; dims=1)) .< 1e-10)
    @test all(abs.(mean(rbm_.vis.θ; dims=1)) .< 1e-10)

    ω = 1 ./ sqrt.(sum(zerosum(rbm.weights).^2; dims = vdims(rbm)))
    Δw = mean(rbm.weights; dims=1)
    rbm_ = RBMs.gauge(deepcopy(rbm))
    @test rbm_.vis.θ ≈ zerosum(rbm.visible.θ)
    @test rbm_.hid.θ ≈ rbm.hidden.θ
    @test rbm_.hid.γ ≈ rbm.hidden.γ
    @test rbm_.weights ≈ zerosum(rbm.weights) .* ω
    ω = dropdims(ω; dims=vdims(rbm))
    rbm_.hid.θ .+= RBMs.sum_(Δw; dims = vdims(rbm))
    rbm_.hid.θ .*= ω
    rbm_.hid.γ .*= ω .* ω
    I = randn(n..., B...)
    v1 = RBMs.sample_from_inputs(rbm.visible, +I)
    v2 = RBMs.sample_from_inputs(rbm.visible, -I)
    Δf = free_energy(rbm,  v1) - free_energy(rbm,  v2)
    Δg = free_energy(rbm_, v1) - free_energy(rbm_, v2)
    @test Δf ≈ Δg
end

@testset "gauge gradient" begin
    n = (5,2,3,5)
    m = (4,3,2)
    B = (3,1)

    rbm = RBM(Potts(n...), ReLU(m...))
    randn!(rbm.weights)
    randn!(rbm.visible.θ)
    randn!(rbm.hidden.θ)
    rand!(rbm.hidden.γ)

    fun(rbm) = mean(sin.(rbm.weights)) + mean(cos.(rbm.visible.θ)) + mean(tan.(rbm.hidden.θ)) + mean(tanh.(rbm.hidden.γ))
    ps = Flux.params(rbm)
    gs = gradient(ps) do
        fun(gauge(rbm))
    end
    eg = randn(size(rbm.visible.θ))
    eθ = randn(size(rbm.hidden.θ))
    eγ = randn(size(rbm.hidden.γ))
    ew = randn(size(rbm.weights))
    @test central_fdm(5,1)(0) do ϵ
        vis = Potts(rbm.visible.θ .+ ϵ .* eg)
        hid = ReLU(rbm.hidden.θ .+ ϵ .* eθ, rbm.hidden.γ .+ ϵ .* eγ)
        w = rbm.weights .+ ϵ .* ew
        fun(gauge(RBM(vis, hid, w)))
    end ≈ sum(sum(gs[p] .* e) for (p,e) in zip(ps, (eg, eθ, eγ, ew)))
end

@testset "gauge grads 2" begin
    n = (5,2,3,5)
    m = (4,3,2)
    B = (3,1)

    # grads are perpendicular to gauge surface
    rbm = RBM(Potts(n...), ReLU(m...))
    randn!(rbm.weights)
    randn!(rbm.visible.θ)
    randn!(rbm.hidden.θ)
    rand!(rbm.hidden.γ)

    ps = Flux.params(rbm)
    v = RBMs.sample_from_inputs(rbm.visible)
    gs = gradient(ps) do
        rbm_ = gauge(rbm)
        free_energy(rbm_, v) + jerome_regularization(rbm_)
    end
    # weights scale
    @test norm(sum(gs[rbm.weights] .* rbm.weights, dims=vdims(rbm))) ≤ 1e-10
    # zero sum
    @test norm(sum(gs[rbm.weights]; dims=1)) ≤ 1e-10
    @test norm(sum(gs[rbm.visible.θ]; dims=1)) ≤ 1e-10
end

@testset "in-place gauge!" begin
    rbm = RBM(Binary(n...), Gaussian(m...))
    randn!(rbm.weights)
    randn!(rbm.visible.θ)
    randn!(rbm.hidden.θ)
    rand!(rbm.hidden.γ)
    rbm_ = gauge(rbm)
    gauge!(rbm)
    @test rbm.weights ≈ rbm_.weights
    @test rbm.visible.θ ≈ rbm_.vis.θ
    @test rbm.hidden.θ ≈ rbm_.hid.θ
    @test rbm.hidden.γ ≈ rbm_.hid.γ

    rbm = RBM(Potts(n...), ReLU(m...))
    randn!(rbm.weights)
    randn!(rbm.visible.θ)
    randn!(rbm.hidden.θ)
    rand!(rbm.hidden.γ)
    rbm_ = gauge(rbm)
    gauge!(rbm)
    @test rbm.weights ≈ rbm_.weights
    @test rbm.visible.θ ≈ rbm_.vis.θ
    @test rbm.hidden.θ ≈ rbm_.hid.θ
    @test rbm.hidden.γ ≈ rbm_.hid.γ
end

@testset "gauge gradient, in-place" begin
    n = (5,2,3,5)
    m = (4,3,2)
    B = (3,1)
    fun(rbm) = mean(sin.(rbm.weights)) + mean(cos.(rbm.visible.θ)) + mean(tan.(rbm.hidden.θ)) + mean(tanh.(rbm.hidden.γ))

    rbm = RBM(Binary(n...), ReLU(m...))
    randn!(rbm.weights)
    randn!(rbm.visible.θ)
    randn!(rbm.hidden.θ)
    rand!(rbm.hidden.γ)
    gauge!(rbm)
    ps = Flux.params(rbm)
    gs = gradient(ps) do
        fun(rbm)
    end
    gauge!(gs, rbm)
    gs_ = gradient(ps) do
        fun(gauge(rbm))
    end
    @test gs[rbm.weights] ≈ gs_[rbm.weights]
    @test gs[rbm.visible.θ] ≈ gs_[rbm.visible.θ]
    @test gs[rbm.hidden.θ] ≈ gs_[rbm.hidden.θ]
    @test gs[rbm.hidden.γ] ≈ gs_[rbm.hidden.γ]

    rbm = RBM(Potts(n...), ReLU(m...))
    randn!(rbm.weights)
    randn!(rbm.visible.θ)
    randn!(rbm.hidden.θ)
    rand!(rbm.hidden.γ)
    gauge!(rbm)
    ps = Flux.params(rbm)
    gs = gradient(ps) do
        fun(rbm)
    end
    gs_ = gradient(ps) do
        fun(gauge(rbm))
    end
    gauge!(gs, rbm)
    @test gs[rbm.weights] ≈ gs_[rbm.weights]
    @test gs[rbm.visible.θ] ≈ gs_[rbm.visible.θ]
    @test gs[rbm.hidden.θ] ≈ gs_[rbm.hidden.θ]
    @test gs[rbm.hidden.γ] ≈ gs_[rbm.hidden.γ]
end
