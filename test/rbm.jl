using Test: @test, @testset
import Statistics
import Random
import LinearAlgebra
import Zygote
import QuadGK
import RestrictedBoltzmannMachines as RBMs

@testset "batches, n=$n, m=$m, Bv=$Bv, Bh=$Bh" for n in (5, (5,2)), m in (2, (3,4)), Bv in ((), (3,2)), Bh in ((), (3,2))
    rbm = RBMs.BinaryRBM(randn(n...), randn(m...), randn(n..., m...))
    wmat = reshape(rbm.w, length(rbm.visible), length(rbm.hidden))
    v = Random.bitrand(n..., Bv...)
    h = Random.bitrand(m..., Bh...)

    @test RBMs.batch_size(rbm.visible, v) == Bv
    @test RBMs.batch_size(rbm.hidden, h) == Bh

    @test size(RBMs.inputs_v_to_h(rbm, v)) == (size(rbm.hidden)...,  Bv...)
    @test size(RBMs.inputs_h_to_v(rbm, h)) == (size(rbm.visible)..., Bh...)

    if length(Bv) == length(Bh) == 0
        @test RBMs.interaction_energy(rbm, v, h) isa Number
        @test RBMs.interaction_energy(rbm, v, h) ≈ -vec(v)' * wmat * vec(h)
        @test RBMs.energy(rbm, v, h) isa Number
    elseif length(Bv) == 0
        hmat = reshape(h, length(rbm.hidden), :)
        E = -vec(v)' * wmat * hmat
        @test RBMs.interaction_energy(rbm, v, h) isa AbstractArray
        @test size(RBMs.interaction_energy(rbm, v, h)) == Bh
        @test RBMs.interaction_energy(rbm, v, h) ≈ reshape(E, Bh)
        @test size(RBMs.energy(rbm, v, h)) == Bh
    elseif length(Bh) == 0
        vmat = reshape(v, length(rbm.visible), :)
        E = -vec(h)' * wmat' * vmat
        @test RBMs.interaction_energy(rbm, v, h) isa AbstractArray
        @test size(RBMs.interaction_energy(rbm, v, h)) == Bv
        @test RBMs.interaction_energy(rbm, v, h) ≈ reshape(E, Bv)
        @test size(RBMs.energy(rbm, v, h)) == Bv
    else
        vmat = reshape(v, length(rbm.visible), :)
        hmat = reshape(h, length(rbm.hidden), :)
        E = -LinearAlgebra.dot.(eachcol(vmat), Ref(wmat), eachcol(hmat))
        @test RBMs.interaction_energy(rbm, v, h) isa AbstractArray
        @test size(RBMs.interaction_energy(rbm, v, h)) == Bv == Bh
        @test RBMs.interaction_energy(rbm, v, h) ≈ reshape(E, Bv)
        @test size(RBMs.energy(rbm, v, h)) == Bv == Bh
    end
    @inferred RBMs.inputs_v_to_h(rbm, v)
    @inferred RBMs.inputs_h_to_v(rbm, h)
    @inferred RBMs.interaction_energy(rbm, v, h)
    @inferred RBMs.energy(rbm, v, h)
    gs = Zygote.gradient(rbm) do rbm
        Statistics.mean(RBMs.energy(rbm, v, h))
    end
    ∂w = @inferred RBMs.∂interaction_energy(rbm, v, h)
    @test ∂w ≈ only(gs).w
end

@testset "sample_v_from_v and sample_h_from_h on binary RBM" begin
    rbm = RBMs.BinaryRBM(randn(3,2), randn(2,3), zeros(3,2,2,3))
    v = Random.bitrand(size(rbm.visible)..., 10^6)
    v = RBMs.sample_v_from_v(rbm, v)
    @test RBMs.batchmean(rbm.visible, v) ≈ RBMs.transfer_mean(rbm.visible) rtol=0.1

    h = Random.bitrand(size(rbm.hidden)...,  10^6)
    h = RBMs.sample_h_from_h(rbm, h)
    @test RBMs.batchmean(rbm.hidden, h) ≈ RBMs.transfer_mean(rbm.hidden) rtol=0.1

    Random.randn!(rbm.w)
    h = RBMs.sample_h_from_v(rbm, v)
    μ = RBMs.transfer_mean(rbm.hidden, RBMs.inputs_v_to_h(rbm, v))
    @test RBMs.batchmean(rbm.hidden, h) ≈ RBMs.batchmean(rbm.hidden, μ) rtol=0.1

    Random.randn!(rbm.w)
    v = RBMs.sample_v_from_h(rbm, h)
    μ = RBMs.transfer_mean(rbm.visible, RBMs.inputs_h_to_v(rbm, h))
    @test RBMs.batchmean(rbm.visible, v) ≈ RBMs.batchmean(rbm.visible, μ) rtol=0.1
end

@testset "rbm convenience constructors" begin
    rbm = RBMs.BinaryRBM(randn(5), randn(3), randn(5,3))
    @test rbm.visible isa RBMs.Binary
    @test rbm.hidden isa RBMs.Binary
    @test size(rbm.w) == (5,3)

    rbm = RBMs.HopfieldRBM(randn(5), randn(3), rand(3), randn(5,3))
    @test rbm.visible isa RBMs.Spin
    @test rbm.hidden isa RBMs.Gaussian
    @test size(RBMs.weights(rbm)) == (5,3)
    @test all(rbm.hidden.γ .> 0)

    rbm = RBMs.HopfieldRBM(randn(5), randn(5,3))
    @test RBMs.visible(rbm) isa RBMs.Spin
    @test iszero(RBMs.transfer_mean(RBMs.hidden(rbm)))
    @test iszero(RBMs.transfer_var(RBMs.hidden(rbm)) .- 1)
    @test size(RBMs.weights(rbm)) == (5,3)
end

@testset "Binary-Binary RBM" begin
    rbm = RBMs.BinaryRBM(randn(5,2), randn(4,3), randn(5,2,4,3))
    v = rand(Bool, size(rbm.visible)..., 7)
    h = rand(Bool, size(rbm.hidden)..., 7)
    @test size(@inferred RBMs.interaction_energy(rbm, v, h)) == (7,)
    @test size(@inferred RBMs.energy(rbm, v, h)) == (7,)

    Ew = -[sum(v[i,j,b] * rbm.w[i,j,μ,ν] * h[μ,ν,b] for i=1:5, j=1:2, μ=1:4, ν=1:3) for b=1:7]
    @test RBMs.interaction_energy(rbm, v, h) ≈ Ew
    @test RBMs.energy(rbm, v, h) ≈ RBMs.energy(rbm.visible, v) + RBMs.energy(rbm.hidden, h) + Ew

    @test (@inferred RBMs.energy(rbm, v[:,:,1], h[:,:,1])) isa Real
    @test (@inferred RBMs.energy(rbm, v[:,:,1], h)) isa AbstractVector{<:Real}
    @test (@inferred RBMs.energy(rbm, v, h[:,:,1])) isa AbstractVector{<:Real}

    for b = 1:7
        @test RBMs.energy(rbm, v[:,:,b], h[:,:,b]) ≈ RBMs.energy(rbm, v, h)[b]
        @test RBMs.energy(rbm, v[:,:,b], h) ≈ [RBMs.energy(rbm, v[:,:,b], h[:,:,k]) for k=1:7]
        @test RBMs.energy(rbm, v, h[:,:,b]) ≈ [RBMs.energy(rbm, v[:,:,k], h[:,:,b]) for k=1:7]
    end

    @test size(@inferred RBMs.sample_h_from_v(rbm, v)) == size(h)
    @test size(@inferred RBMs.sample_v_from_h(rbm, h)) == size(v)
    @test size(@inferred RBMs.sample_h_from_v(rbm, v[:,:,1])) == size(rbm.hidden)
    @test size(@inferred RBMs.sample_v_from_h(rbm, h[:,:,1])) == size(rbm.visible)
    for k = 1:3
        @test size(@inferred RBMs.sample_v_from_v(rbm, v; steps=k)) == size(v)
        @test size(@inferred RBMs.sample_h_from_h(rbm, h; steps=k)) == size(h)
        @test size(@inferred RBMs.sample_v_from_v(rbm, v[:,:,1]; steps=k)) == size(rbm.visible)
        @test size(@inferred RBMs.sample_h_from_h(rbm, h[:,:,1]; steps=k)) == size(rbm.hidden)
    end

    @test size(@inferred RBMs.mean_h_from_v(rbm, v)) == size(h)
    @test size(@inferred RBMs.mean_v_from_h(rbm, h)) == size(v)
    @test size(@inferred RBMs.mean_h_from_v(rbm, v[:,:,1])) == size(rbm.hidden)
    @test size(@inferred RBMs.mean_v_from_h(rbm, h[:,:,1])) == size(rbm.visible)

    @test size(@inferred RBMs.mode_h_from_v(rbm, v)) == size(h)
    @test size(@inferred RBMs.mode_v_from_h(rbm, h)) == size(v)
    @test size(@inferred RBMs.mode_h_from_v(rbm, v[:,:,1])) == size(rbm.hidden)
    @test size(@inferred RBMs.mode_v_from_h(rbm, h[:,:,1])) == size(rbm.visible)

    @test size(@inferred RBMs.var_h_from_v(rbm, v)) == size(h)
    @test size(@inferred RBMs.var_v_from_h(rbm, h)) == size(v)
    @test size(@inferred RBMs.var_h_from_v(rbm, v[:,:,1])) == size(rbm.hidden)
    @test size(@inferred RBMs.var_v_from_h(rbm, h[:,:,1])) == size(rbm.visible)

    @test size(@inferred RBMs.free_energy(rbm, v)) == (7,)
    @test size(@inferred RBMs.reconstruction_error(rbm, v)) == (7,)
    @test (@inferred RBMs.free_energy(rbm, v[:,:,1])) isa Real
    @test (@inferred RBMs.reconstruction_error(rbm, v[:,:,1])) isa Real

    @inferred RBMs.mirror(rbm)
    @test RBMs.mirror(rbm).visible == rbm.hidden
    @test RBMs.mirror(rbm).hidden == rbm.visible
    @test RBMs.energy(RBMs.mirror(rbm), h, v) ≈ RBMs.energy(rbm, v, h)
end

@testset "Gaussian-Gaussian RBM, 1-dimension" begin
    rbm = RBMs.RBM(
        RBMs.Gaussian(randn(1), rand(1) .+ 1),
        RBMs.Gaussian(randn(1), rand(1) .+ 1),
        randn(1, 1) / 1e2
    )

    @assert LinearAlgebra.isposdef([
        rbm.visible.γ -rbm.w;
        -rbm.w'  rbm.hidden.γ
    ])

    @test RBMs.log_partition(rbm; β = 1) ≈ RBMs.log_partition(rbm)

    Z, ϵ = QuadGK.quadgk(x -> exp(-only(RBMs.free_energy(rbm, [x;;]))), -Inf, Inf)
    @test RBMs.log_partition(rbm) ≈ log(Z)

    β = 1.5
    Z, ϵ = QuadGK.quadgk(x -> exp(-β * only(RBMs.free_energy(rbm, [x;;]; β))), -Inf, Inf)
    @test RBMs.log_partition(rbm; β) ≈ log(Z)
end

@testset "Gaussian-Gaussian RBM, multi-dimensional" begin
    n = (10, 3)
    m = (7, 2)
    rbm = RBMs.RBM(
        RBMs.Gaussian(randn(n...), rand(n...) .+ 0.5),
        RBMs.Gaussian(randn(m...), rand(m...) .+ 0.5),
        randn(n..., m...) / (10 * prod(n) * prod(m))
    )

    N = length(rbm.visible)
    M = length(rbm.hidden)

    θ = [
        vec(rbm.visible.θ);
        vec(rbm.hidden.θ)
    ]
    γv = vec(rbm.visible.γ)
    γh = vec(rbm.hidden.γ)
    w = reshape(rbm.w, length(rbm.visible), length(rbm.hidden))
    A = [LinearAlgebra.diagm(γv) -w;
         -w'  LinearAlgebra.diagm(γh)]

    v = randn(n..., 1)
    h = randn(m..., 1)
    x = [reshape(v, N, 1); reshape(h, M, 1)]

    @test RBMs.energy(rbm, v, h) ≈ x' * A * x / 2 - θ' * x
    @test RBMs.log_partition(rbm) ≈ (
        (N + M)/2 * log(2π) + θ' * inv(A) * θ / 2 - LinearAlgebra.logdet(A)/2
    )
    @test RBMs.log_partition(rbm; β = 1) ≈ RBMs.log_partition(rbm)

    β = rand()
    @test RBMs.log_partition(rbm; β) ≈ (
        (N + M)/2 * log(2π) + β * θ' * inv(A) * θ / 2 - LinearAlgebra.logdet(β*A)/2
    ) rtol=1e-6

    @test RBMs.log_likelihood(rbm, v; β = 1) ≈ RBMs.log_likelihood(rbm, v)
    @test RBMs.log_likelihood(rbm, v; β) ≈ (
        -β * RBMs.free_energy(rbm, v; β) .- RBMs.log_partition(rbm; β)
    )

    Ev = sum(@. rbm.visible.γ * v^2 / 2 - rbm.visible.θ * v)
    Fv = sum(-(rbm.hidden.θ .+ RBMs.inputs_v_to_h(rbm, v)).^2 ./ 2rbm.hidden.γ)
    @test only(RBMs.free_energy(rbm, v)) ≈ Ev + Fv - sum(log.(2π ./ rbm.hidden.γ)) / 2

    gs = Zygote.gradient(rbm) do rbm
        RBMs.log_partition(rbm)
    end
    ∂θv = vec(only(gs).visible.θ)
    ∂θh = vec(only(gs).hidden.θ)
    ∂γv = vec(only(gs).visible.γ)
    ∂γh = vec(only(gs).hidden.γ)
    ∂w = reshape(only(gs).w, length(rbm.visible), length(rbm.hidden))

    Σ = inv(A) # covariances
    μ = Σ * θ  # means
    C = Σ + μ * μ' # non-centered second moments

    @test [∂θv; ∂θh] ≈ μ
    @test -2∂γv ≈ LinearAlgebra.diag(C[1:N, 1:N]) # <vi^2>
    @test -2∂γh ≈ LinearAlgebra.diag(C[(N + 1):end, (N + 1):end]) # <hμ^2>
    @test ∂w ≈ C[1:N, (N + 1):end] # <vi*hμ>
end
