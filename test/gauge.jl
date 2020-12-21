using Random, Test, Statistics, Zygote, Flux, FiniteDifferences, LinearAlgebra
using RestrictedBoltzmannMachines, OneHot
using RestrictedBoltzmannMachines: sumdrop, meandrop,
    gauge, zerosum, rescale,
    gauge!, zerosum!, rescale!,
    gauge!, rescale!

n = (5,2,3,5)
m = (4,3,2)
B = (3,1)

@testset "zerosum" begin
    rbm = RBM(Potts(5,6), Gaussian(3))
    rand!(rbm.vis.θ); rand!(rbm.weights);
    rand!(rbm.hid.θ); rand!(rbm.hid.γ);
    
    rbm.vis.θ .+= 1
    @test norm(sum(rbm.vis.θ; dims=1)) > 1
    zerosum!(rbm.vis)
    @test norm(sum(rbm.vis.θ; dims=1)) < 1e-10

    rbm.weights .+= 1
    @test norm(sum(rbm.weights); dims=1) > 1
    zerosum!(rbm)
    @test norm(sum(rbm.weights); dims=1) < 1e-10
end

@testset "cd gauge invariance Binary / Gaussian" begin
    rbm = RBM(Binary(5), Gaussian(3))
    randn!(rbm.vis.θ); randn!(rbm.weights);
    randn!(rbm.hid.θ); rand!(rbm.hid.γ);
    vd = random(rbm.vis, zeros(size(rbm.vis)..., 10))
    vm = random(rbm.vis, zeros(size(rbm.vis)..., 10))
    λ, κ = rand(size(rbm.hid)...), randn(size(rbm.hid)...)
    dλ, dκ = gradient(λ, κ) do λ, κ
        g_ = rbm.vis.θ .+ tensormul_lf(rbm.weights, κ, Val(ndims(rbm.hid)))
        w_ = rbm.weights ./ reshape(λ, ones(Int, ndims(rbm.vis))..., size(rbm.hid)...)
        θ_ = (rbm.hid.θ .- rbm.hid.γ .* κ) ./ λ
        γ_ = rbm.hid.γ ./ λ.^2
        rbm_ = RBM(Binary(g_), Gaussian(θ_, γ_), w_)
        contrastive_divergence_v(rbm_, vd, vm)
    end
    @test norm(dλ) ≤ 1e-10
    @test norm(dκ) ≤ 1e-10
end

@testset "gauge Binary / ReLU" begin
    rbm = RBM(Binary(n...), Gaussian(m...))
    randn!(rbm.weights)
    randn!(rbm.vis.θ)
    randn!(rbm.hid.θ)
    rand!(rbm.hid.γ)

    rbm_ = gauge(rbm)
    @test gauge(rbm_).weights ≈ rbm_.weights
    @test all(sum(rbm_.weights.^2; dims=vdims(rbm)) .≈ 1)

    #= Tests that the machine did not lose expressivity. More precisely,
    compensante rescaling by scaling of hidden units and show that this
    restores original free energies. =#
    rbm_ = rescale(deepcopy(rbm))
    ω = 1 ./ sqrt.(sum(rbm.weights.^2; dims=vdims(rbm)))
    @test rbm_.weights ≈ rbm.weights .* ω
    @test rbm_.vis.θ ≈ rbm.vis.θ
    @test rbm_.hid.θ ≈ rbm.hid.θ
    @test rbm_.hid.γ ≈ rbm.hid.γ
    ω = dropdims(ω; dims = vdims(rbm))
    @test size(ω) == size(rbm.hid)
    rbm_.hid.θ .*= ω
    rbm_.hid.γ .*= ω .* ω
    I = randn(n..., B...)
    v1 = random(rbm.vis, +I)
    v2 = random(rbm.vis, -I)
    Δf = free_energy_v(rbm,  v1) - free_energy_v(rbm,  v2)
    Δg = free_energy_v(rbm_, v1) - free_energy_v(rbm_, v2)
    @test Δf ≈ Δg
end

@testset "gauge Onehot / ReLU" begin
    rbm = RBM(Potts(n...), ReLU(m...))
    randn!(rbm.weights)
    randn!(rbm.vis.θ)
    randn!(rbm.hid.θ)
    rand!(rbm.hid.γ)

    rbm_ = RBMs.gauge(rbm)
    @test RBMs.gauge(rbm_).weights ≈ rbm_.weights
    @test RBMs.gauge(rbm_).vis.θ ≈ rbm_.vis.θ
    @test all(sum(rbm_.weights.^2; dims=vdims(rbm)) .≈ 1)
    @test all(abs.(mean(rbm_.weights; dims=1)) .< 1e-10)
    @test all(abs.(mean(rbm_.vis.θ; dims=1)) .< 1e-10)

    ω = 1 ./ sqrt.(sum(zerosum(rbm.weights).^2; dims = vdims(rbm)))
    Δw = mean(rbm.weights; dims=1)
    rbm_ = RBMs.gauge(deepcopy(rbm))
    @test rbm_.vis.θ ≈ zerosum(rbm.vis.θ)
    @test rbm_.hid.θ ≈ rbm.hid.θ
    @test rbm_.hid.γ ≈ rbm.hid.γ
    @test rbm_.weights ≈ zerosum(rbm.weights) .* ω
    ω = dropdims(ω; dims=vdims(rbm))
    rbm_.hid.θ .+= sumdrop(Δw; dims = vdims(rbm))
    rbm_.hid.θ .*= ω
    rbm_.hid.γ .*= ω .* ω
    I = randn(n..., B...)
    v1 = random(rbm.vis, +I)
    v2 = random(rbm.vis, -I)
    Δf = free_energy_v(rbm,  v1) - free_energy_v(rbm,  v2)
    Δg = free_energy_v(rbm_, v1) - free_energy_v(rbm_, v2)
    @test Δf ≈ Δg
end

@testset "gauge gradient" begin
    n = (5,2,3,5)
    m = (4,3,2)
    B = (3,1)

    rbm = RBM(Potts(n...), ReLU(m...))
    randn!(rbm.weights)
    randn!(rbm.vis.θ)
    randn!(rbm.hid.θ)
    rand!(rbm.hid.γ)

    fun(rbm) = mean(sin.(rbm.weights)) + mean(cos.(rbm.vis.θ)) + mean(tan.(rbm.hid.θ)) + mean(tanh.(rbm.hid.γ))
    ps = Flux.params(rbm)
    gs = gradient(ps) do
        fun(gauge(rbm))
    end
    eg = randn(size(rbm.vis.θ))
    eθ = randn(size(rbm.hid.θ))
    eγ = randn(size(rbm.hid.γ))
    ew = randn(size(rbm.weights))
    @test central_fdm(5,1)(0) do ϵ
        vis = Potts(rbm.vis.θ .+ ϵ .* eg)
        hid = ReLU(rbm.hid.θ .+ ϵ .* eθ, rbm.hid.γ .+ ϵ .* eγ)
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
    randn!(rbm.vis.θ)
    randn!(rbm.hid.θ)
    rand!(rbm.hid.γ)

    ps = params(rbm)
    v = random(rbm.vis)
    gs = gradient(ps) do
        rbm_ = gauge(rbm)
        free_energy_v(rbm_, v) + jerome_regularization(rbm_)
    end
    # weights scale
    @test norm(sum(gs[rbm.weights] .* rbm.weights, dims=vdims(rbm))) ≤ 1e-10
    # zero sum
    @test norm(sum(gs[rbm.weights]; dims=1)) ≤ 1e-10
    @test norm(sum(gs[rbm.vis.θ]; dims=1)) ≤ 1e-10
end

@testset "in-place gauge!" begin
    rbm = RBM(Binary(n...), Gaussian(m...))
    randn!(rbm.weights)
    randn!(rbm.vis.θ)
    randn!(rbm.hid.θ)
    rand!(rbm.hid.γ)
    rbm_ = gauge(rbm)
    gauge!(rbm)
    @test rbm.weights ≈ rbm_.weights
    @test rbm.vis.θ ≈ rbm_.vis.θ
    @test rbm.hid.θ ≈ rbm_.hid.θ
    @test rbm.hid.γ ≈ rbm_.hid.γ

    rbm = RBM(Potts(n...), ReLU(m...))
    randn!(rbm.weights)
    randn!(rbm.vis.θ)
    randn!(rbm.hid.θ)
    rand!(rbm.hid.γ)
    rbm_ = gauge(rbm)
    gauge!(rbm)
    @test rbm.weights ≈ rbm_.weights
    @test rbm.vis.θ ≈ rbm_.vis.θ
    @test rbm.hid.θ ≈ rbm_.hid.θ
    @test rbm.hid.γ ≈ rbm_.hid.γ
end

@testset "gauge gradient, in-place" begin
    n = (5,2,3,5)
    m = (4,3,2)
    B = (3,1)
    fun(rbm) = mean(sin.(rbm.weights)) + mean(cos.(rbm.vis.θ)) + mean(tan.(rbm.hid.θ)) + mean(tanh.(rbm.hid.γ))

    rbm = RBM(Binary(n...), ReLU(m...))
    randn!(rbm.weights)
    randn!(rbm.vis.θ)
    randn!(rbm.hid.θ)
    rand!(rbm.hid.γ)
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
    @test gs[rbm.vis.θ] ≈ gs_[rbm.vis.θ]
    @test gs[rbm.hid.θ] ≈ gs_[rbm.hid.θ]
    @test gs[rbm.hid.γ] ≈ gs_[rbm.hid.γ]

    rbm = RBM(Potts(n...), ReLU(m...))
    randn!(rbm.weights)
    randn!(rbm.vis.θ)
    randn!(rbm.hid.θ)
    rand!(rbm.hid.γ)
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
    @test gs[rbm.vis.θ] ≈ gs_[rbm.vis.θ]
    @test gs[rbm.hid.θ] ≈ gs_[rbm.hid.θ]
    @test gs[rbm.hid.γ] ≈ gs_[rbm.hid.γ]
end

# @testset "center" begin
#     rbm = RBM(Binary(5), Gaussian(3))
#     randn!(rbm.weights); randn!(rbm.vis.θ); randn!(rbm.hid.θ); randn!(rbm.hid.γ)
#     rbm_ = RBMs.center(rbm)
#     @test rbm_.vis.θ ≈ rbm.vis.θ
#     @test rbm_.hid.θ ≈ rbm.hid.θ
#     @test rbm_.hid.γ ≈ rbm.hid.γ
#     @test norm(rbm.vis.θ' * rbm_.weights ) ≤ 1e-10
#     @test rbm.vis.θ' * (rbm.weights .- rbm_.weights) ≈ rbm.vis.θ' * rbm.weights
# end
#
# @testset "center vs center!" begin
#     rbm = RBM(Binary(5), Gaussian(3))
#     randn!(rbm.weights); randn!(rbm.vis.θ); randn!(rbm.hid.θ); randn!(rbm.hid.γ)
#     rbm_ = RBMs.center(rbm)
#     RBMs.center!(rbm)
#     @test rbm_.weights ≈ rbm.weights
#     @test rbm_.vis.θ ≈ rbm.vis.θ
#     @test rbm_.hid.θ ≈ rbm.hid.θ
#     @test rbm_.hid.γ ≈ rbm.hid.γ
# end
#
# @testset "center!" begin
# end
#
# rbm = RBM(Binary(5), Gaussian(3))
# randn!(rbm.weights); randn!(rbm.vis.θ)
# RBMs.center!(rbm)
# ps = params(rbm)
# gs = Zygote.gradient(ps) do
#     sum(sin.(rbm.weights)) + sum(cos.(rbm.vis.θ))
# end
# dw0 = copy(gs[rbm.weights]); dg0 = copy(gs[rbm.vis.θ]);
# RBMs.center!(gs, rbm)
# @test norm(rbm.vis.θ' * gs[rbm.weights] + gs[rbm.vis.θ]' * rbm.weights) ≤ 1e-10
# @test rbm.vis.θ' * dw0 + dg0' * rbm.weights ≈ rbm.vis.θ' * (dw0 .- gs[rbm.weights]) + (dg0 .- gs[rbm.vis.θ])' * rbm.weights
#
# rbm.vis.θ' * gs[rbm.weights] + gs[rbm.vis.θ]' * rbm.weights
# rbm.vis.θ
#
# A = [reshape(rbm.weights, length(g), :)' kron(I(length(rbm.hid)), vec(rbm.vis.θ)')]
# [dotcos([vec(gs[rbm.weights]); vec(gs[rbm.vis.θ])], row) for row in eachrow(A)]
#
# [dotcos([vec(dw0 .- gs[rbm.weights]); vec(dg0 .- gs[rbm.vis.θ])], row) for row in eachrow(A)]
# [dotcos([vec(gs[rbm.weights]); vec(gs[rbm.vis.θ])], row) for row in eachrow(A)]


#norm.(eachrow(A))

# dotcos([vec(dw0 .- gs[rbm.weights]); vec(dg0 .- gs[rbm.vis.θ])],
#          [vec(rbm.vis.θ); vec(rbm.weights)])
# dotcos([vec(gs[rbm.weights]); vec(gs[rbm.vis.θ])],
#          [vec(rbm.vis.θ); vec(rbm.weights)])




#
# @testset "center" begin
#
# end
#
# rbm0 = deepcopy(rbm)
#
# [dotcos(rbm.weights[:,μ] .- RBMs.center(rbm).weights[:,μ], rbm.vis.θ) for μ = 1:3]
# [dotcos(RBMs.center(rbm).weights[:,μ], rbm.vis.θ) for μ = 1:3]
# norm(RBMs.center(rbm).weights)
#
#
#
# rbm = RBM(Binary(5), Gaussian(3))
# randn!(rbm.weights); randn!(rbm.vis.θ);
# RBMs.center!(rbm)
#
#
# rbm = RBM(Binary(5), Gaussian(3))
# randn!(rbm.vis.θ)
# randn!(rbm.weights)
# RBMs.center!(rbm)
# ps = params(rbm.weights)
# gs_ = Zygote.gradient(ps) do
#     sum(sin.(RBMs.center(rbm).weights))
# end
# gs = Zygote.gradient(ps) do
#     sum(sin.(rbm.weights))
# end
#
# gs[rbm.weights]
#
#
#



#gs_[rbm.weights]



# @testset "center!" begin
#     dw0 = copy(gs[rbm.weights]); dg0 = copy(gs[rbm.vis.θ]);
#     rbm.vis.θ' * dw0 + dg0' * rbm.weights
#     rbm.vis.θ' * gs[rbm.weights] + gs[rbm.vis.θ]' * rbm.weights
#     rbm.vis.θ' * (dw0 .- gs[rbm.weights]) + (dg0 .- gs[rbm.vis.θ])' * rbm.weights
# end

#
# gs[rbm.vis.θ]
#
# RBMs.center!(rbm)
# sum(rbm.weights .* rbm.vis.θ; dims=(1,2))
# sum(rbm.weights .* rbm.vis.θ; dims=(1,2))
#
#
# sum(RBMs.center(rbm).weights .* rbm.vis.θ; dims=(1,2))
