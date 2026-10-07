#= Regression tests for the PTT machinery, each guarding a property whose failure was observed
while training binary-hidden RBMs on protein and RNA alignments with `ptt!`. They complement
`ptt.jl`, which checks the sampler and the ladder against exact enumeration. =#

import Random
import RestrictedBoltzmannMachines as RBMs
using Test: @test, @testset
using Optimisers: Nesterov, setup, update!
using RestrictedBoltzmannMachines: RBM, BinaryRBM, Binary, Potts, PottsGumbel, StandardizedRBM,
    TrajectoryLadder, CossimDescent, ptt!, initialize!, free_energy, batchmean

Random.seed!(43)

#= An exchange swaps a chain and a reservoir member together with their free energies, so
that `reservoir_F` stays the free energy of the reservoir under the last checkpoint. The
members proposed for exchange are distinct, so no reservoir member is written twice. =#
@testset "exchange bookkeeping" begin
    base = BinaryRBM(randn(6) / 2, randn(3) / 2, randn(6, 3) / 2)
    model = BinaryRBM(base.visible.θ .+ 0.3, base.hidden.θ, base.w .+ randn(6, 3) / 4)
    ladder = TrajectoryLadder(base; nchains = 50, nreservoir = 200, anneal = 20)
    for _ in 1:5
        reservoir₀ = copy(ladder.reservoir)
        chains₀ = copy(ladder.chains)
        proposal = RBMs._propose_exchange(ladder, model)
        @test allunique(proposal.idx)
        @test proposal.Fx ≈ free_energy(base, chains₀)
        @test proposal.Fθx ≈ free_energy(model, chains₀)
        accept = RBMs._exchange!(ladder, proposal)
        @test length(accept) == 50
        @test ladder.reservoir_F ≈ free_energy(base, ladder.reservoir)
        for (n, i) in enumerate(proposal.idx)
            if accept[n]
                @test ladder.chains[:, n] == reservoir₀[:, i]
                @test ladder.reservoir[:, i] == chains₀[:, n]
            else
                @test ladder.chains[:, n] == chains₀[:, n]
                @test ladder.reservoir[:, i] == reservoir₀[:, i]
            end
        end
        ladder.chains .= RBMs.sample_v_from_v(model, ladder.chains; steps = 1)
    end
end

#= After a checkpoint is frozen, the chains were last sampled from the checkpoint, so their
stored free energies refer to it; the new reservoir and its free energies are consistent;
and the model moves on from the checkpoint, not back to an older one. =#
@testset "state after freezing a checkpoint" begin
    rbm = BinaryRBM(randn(8) / 4, randn(3) / 4, randn(8, 3) / 4)
    ladder = TrajectoryLadder(rbm; nchains = 100, nreservoir = 500, anneal = 20)
    K = length(ladder.checkpoints)
    model = BinaryRBM(rbm.visible.θ .+ 0.4, rbm.hidden.θ, rbm.w)
    @test RBMs._push_checkpoint!(ladder, model; steps = 1)
    @test length(ladder.checkpoints) == K + 1
    checkpoint = last(ladder.checkpoints)
    @test checkpoint.visible.θ == model.visible.θ && checkpoint !== model
    @test ladder.chains_F ≈ free_energy(checkpoint, ladder.chains)
    @test ladder.samples_F ≈ free_energy(checkpoint, ladder.samples)
    @test ladder.reservoir == ladder.samples && ladder.reservoir_F == ladder.samples_F
    @test ladder.since_checkpoint == 0
    @test ladder.τint ≥ 0.5 && ladder.τexp ≥ 0
end

#= `CossimDescent` multiplies η by 1 ± δ according to the sign of the dot product of
successive gradients. A gradient sequence whose sign alternates therefore shrinks η
geometrically: this is how minibatches with anticorrelated noise, such as those of an
epoch with few minibatches, collapsed η by ten orders of magnitude in training. A zero
gradient leaves η unchanged, and η never exceeds ηmax. =#
@testset "CossimDescent learning-rate dynamics" begin
    η₀, ηmax, δ = 0.01, 0.1, 0.1
    x = ones(3)
    g = [1.0, 2.0, 3.0]
    state = setup(CossimDescent(η₀, ηmax, δ), x)
    η(state) = state.state[2]
    state, x = update!(state, x, g) # no previous gradient
    @test η(state) == η₀
    for k in 1:20 # alternating signs: anti-aligned at every step
        state, x = update!(state, x, (-1)^k * g)
        @test η(state) ≈ η₀ * (1 - δ)^k
    end
    @test state.state[1] == (-1)^20 * g # the previous gradient is stored
    η₁ = η(state)
    state, x = update!(state, x, zeros(3)) # zero gradient: neither aligned nor anti-aligned
    @test η(state) == η₁
    @test iszero(state.state[1])
    for _ in 1:200 # aligned at every step: grows, then clamps at ηmax
        state, x = update!(state, x, g)
    end
    @test η(state) == ηmax
end

#= Data weights enter the positive phase. With weights that select one of two subsets of
the data, `ptt!` must fit the first moments of the weighted data, not those of all the data.
Potts visible layers, as in sequence alignments, where the weights correct for phylogeny. =#
@testset "ptt! fits the weighted data moments, $V visible" for V in (Potts, PottsGumbel)
    q, L, n = 3, 4, 400
    data = falses(q, L, 2n)
    for s in 1:2n, i in 1:L
        # the first n samples favour color 1, the last n favour color 2
        data[s ≤ n ? (rand() < 0.7 ? 1 : rand(2:3)) : (rand() < 0.7 ? 2 : rand((1, 3))), i, s] = true
    end
    wts = [fill(1.0, n); fill(1.0e-3, n)]
    rbm = RBM(V((q, L)), Binary((2,)), zeros(q, L, 2))
    initialize!(rbm, data; wts)
    ladder = TrajectoryLadder(rbm; nchains = 200, nreservoir = 1000)
    ptt!(rbm, data; ladder, wts, batchsize = 100, iters = 400, optim = CossimDescent(0.05, 0.2))
    μ_model = batchmean(rbm.visible, RBMs.sample_v_from_v(rbm, ladder.samples; steps = 20))
    μ_weighted = batchmean(rbm.visible, data; wts)
    μ_unweighted = batchmean(rbm.visible, data)
    @test maximum(abs, μ_model - μ_weighted) < 0.05
    @test maximum(abs, μ_model - μ_unweighted) > 0.15
end

#= A plain `RBM` is trained as the equivalent `StandardizedRBM` with fixed zero offsets and
unit scales, so both calls must produce the same model from the same random stream. =#
@testset "ptt!(::RBM) matches ptt!(::StandardizedRBM) with fixed standardization" begin
    data = rand(Bool, 6, 300) .* 1.0
    data[1:3, 1:150] .= 1 # some structure
    results = map(1:2) do k
        Random.seed!(11)
        rbm = BinaryRBM(6, 3)
        initialize!(rbm, data)
        model = k == 1 ? rbm : RBMs.PlainStandardizedRBM(rbm)
        ladder = TrajectoryLadder(model; nchains = 50, nreservoir = 200, anneal = 20)
        ptt!(model, data; ladder, batchsize = 50, iters = 60, optim = Nesterov(0.5, 0.9))
        (rbm.w, rbm.visible.θ, rbm.hidden.θ, ladder.logZ, ladder.chains)
    end
    @test all(results[1] .== results[2])
end

#= A rejected update restores the model to the last checkpoint and halves the learning rate.
It must also forget the momentum, which would otherwise repeat the rejected step from the
restored model. The update of a rejected iteration is thus a Nesterov step with zero velocity
from the checkpoint: a displacement of exactly -(1 + ρ) η ∂, with the halved η. =#
@testset "no stale momentum after a rejection in ptt!" begin
    data = rand(Bool, 8, 500) .* 1.0
    data[1:4, 1:250] .= 1
    rbm = BinaryRBM(8, 4)
    initialize!(rbm, data)
    ladder = TrajectoryLadder(rbm; nchains = 200)
    ρ = 0.9
    checked = Ref(0)
    ptt!(
        rbm, data; ladder, batchsize = 100, iters = 300, optim = Nesterov(5.0, ρ),
        callback = (; rbm, state, ∂, ladder, _...) -> begin
            if ladder.rejections > 0 # this iteration's step started from the restored checkpoint
                η = state.w.rule.eta
                @test rbm.w ≈ last(ladder.checkpoints).w - (1 + ρ) * η * ∂.w
                @test state.w.state ≈ -η * ∂.w # the velocity after this single step
                checked[] += 1
            end
        end,
    )
    @test checked[] > 0
end
