import Random
import RestrictedBoltzmannMachines as RBMs
using Test: @test, @testset, @test_throws
using Statistics: mean
using LogExpFunctions: softmax
using StatsBase: sample, Weights
using EllipsisNotation: (..)
using Optimisers: Adam, Descent, setup, update!
using RestrictedBoltzmannMachines: RBM, BinaryRBM, Binary, Spin, Potts, Gaussian,
    TrajectoryLadder, CossimDescent, ptt!, initialize!, free_energy,
    log_partition, log_likelihood, collect_states, center, standardize

Random.seed!(41)

enumerate_states(layer::Union{Binary, Spin}) = collect_states(layer)

function enumerate_states(layer::Potts)
    q, sites = size(layer)
    states = falses(q, sites, q^sites)
    for (n, colors) in enumerate(Iterators.product(ntuple(Returns(1:q), sites)...))
        for (site, color) in enumerate(colors)
            states[color, site, n] = true
        end
    end
    return states
end

total_variation(p::AbstractVector, q::AbstractVector) = sum(abs, p - q) / 2

function empirical_distribution(x::AbstractArray, states::AbstractArray)
    index = Dict(vec(states[.., n]) => n for n in 1:size(states)[end])
    counts = zeros(size(states)[end])
    for n in 1:size(x)[end]
        counts[index[vec(x[.., n])]] += 1
    end
    return counts / size(x)[end]
end

# upper bound on the plug-in total variation between `n` exact samples of `p` and `p`
tv_noise(p::AbstractVector, n::Int) = sum(sqrt.(p .* (1 .- p) ./ n)) / 2

exact_samples(states::AbstractArray, p::AbstractVector, n::Int) =
    states[.., sample(1:length(p), Weights(p), n)]

# a ladder with a single checkpoint, `base`, without freezing or rejecting updates
function fixed_ladder(model, base, states; nchains::Int, nreservoir::Int)
    chains = exact_samples(states, softmax(-free_energy(model, states)), nchains)
    samples = exact_samples(states, softmax(-free_energy(base, states)), nreservoir)
    samples_F = free_energy(base, samples)
    return TrajectoryLadder(
        model, [base], [0.0], chains, free_energy(model, chains), samples, samples_F,
        copy(samples), copy(samples_F), 1.0, 2, 0, NaN, NaN, 1, 0.0, 0.0
    )
end

random_layer(::Type{L}, sz::Dims) where {L <: Union{Binary, Potts}} = L(; θ = randn(sz...) / 2)

@testset "CossimDescent" begin
    st = setup(CossimDescent(0.1, 0.2, 0.5), [1.0, 2.0])
    x = [1.0, 2.0]
    st, x = update!(st, x, [1.0, 0.0]) # no previous gradient: η = 0.1
    @test x ≈ [0.9, 2.0]
    st, x = update!(st, x, [1.0, 1.0]) # aligned: η = 0.15
    @test x ≈ [0.75, 1.85]
    st, x = update!(st, x, [1.0, 0.0]) # aligned: η = min(0.225, 0.2)
    @test x ≈ [0.55, 1.85]
    st, x = update!(st, x, [-1.0, 0.0]) # anti-aligned: η = 0.1
    @test x ≈ [0.65, 1.85]
    @test_throws ArgumentError CossimDescent(0.1, 0.05)
    @test_throws ArgumentError CossimDescent(0.1, 0.2, 1.0)

    ps = (; a = [1.0], b = [2.0])
    gs = (; a = [1.0], b = [-1.0])
    for (optim, halved) in ((CossimDescent(0.1, 0.2), CossimDescent(0.05, 0.2)), (Adam(0.1), Adam(0.05)), (Descent(0.1), Descent(0.05)))
        state = setup(optim, ps)
        RBMs._halve_learning_rate!(state)
        @test update!(state, deepcopy(ps), gs)[2] == update!(setup(halved, ps), deepcopy(ps), gs)[2]
    end
    state = setup(Adam(0.1), (; a = [1.0], b = 1)) # the non-trainable `b` has an empty state
    RBMs._halve_learning_rate!(state)
    @test state.a.rule.eta == 0.05
end

#= Replica exchange with the reservoir, followed by Gibbs sampling, must leave the joint
distribution of the chains and the reservoir invariant: starting both exactly at
equilibrium, they must stay at equilibrium within Monte-Carlo error. =#
@testset "no drift: PTT update, $V visible, standardized = $standardized" for (V, vsz) in ((Binary, (6,)), (Potts, (3, 3))), standardized in (false, true)
    model = RBM(random_layer(V, vsz), Binary(; θ = randn(3) / 2), randn(vsz..., 3) * 0.6)
    base = RBM(model.visible, model.hidden, model.w .+ randn(vsz..., 3) * 0.3)
    if standardized
        model = standardize(model, rand(vsz...) / 3, rand(3) / 3, 0.5 .+ rand(vsz...), 0.5 .+ rand(3))
        base = standardize(base, rand(vsz...) / 3, rand(3) / 3, 0.5 .+ rand(vsz...), 0.5 .+ rand(3))
    end
    states = enumerate_states(model.visible)
    p = softmax(-free_energy(model, states))
    p₀ = softmax(-free_energy(base, states))
    ladder = fixed_ladder(model, base, states; nchains = 20_000, nreservoir = 40_000)
    for _ in 1:4
        @test RBMs._ptt_update!(ladder, model; steps = 1) === :accepted
        @test total_variation(empirical_distribution(ladder.chains, states), p) < 4tv_noise(p, 20_000)
        @test total_variation(empirical_distribution(ladder.reservoir, states), p₀) < 4tv_noise(p₀, 40_000)
        @test ladder.reservoir_F ≈ free_energy(base, ladder.reservoir)
    end
    @test 0 < ladder.acceptance < 1
end

#= A bimodal Hopfield model, whose Gibbs chains never leave the mode they start in. PTT
equilibrates the relative weights of the two modes through the ladder. Chains moved out of
equilibrium, all into one mode, are exchanged into the reservoir, which is therefore large
enough that this leaves its mode weights unchanged within the tolerance. =#
@testset "PTT mixes between modes" begin
    N = 10
    rbm = RBM(Spin(; θ = fill(0.03, N)), Gaussian(; θ = zeros(1), γ = ones(1)), fill(0.8, N, 1))
    states = enumerate_states(rbm.visible)
    p = softmax(-free_energy(rbm, states))
    up = vec(sum(states; dims = 1) .> 0)
    p_up = sum(p[up]) # exact weight of the mode with positive magnetization
    @test 0.6 < p_up < 0.9

    ladder = TrajectoryLadder(rbm; nchains = 1000, nreservoir = 100_000)
    @test length(ladder.checkpoints) > 2 # needs intermediate models
    @test abs(log_partition(ladder) - log_partition(rbm)) < 0.05
    @test abs(mean(sum(ladder.chains; dims = 1) .> 0) - p_up) < 4sqrt(p_up * (1 - p_up) / 1000)

    ladder.chains .= 1 # all chains in the positive mode
    gibbs = RBMs.sample_v_from_v(rbm, copy(ladder.chains); steps = 200)
    @test all(sum(gibbs; dims = 1) .> 0) # Gibbs sampling stays in that mode
    for _ in 1:100
        RBMs._ptt_update!(ladder, rbm; steps = 1)
    end
    f_up = mean(sum(ladder.chains; dims = 1) .> 0)
    @test abs(f_up - p_up) < 4sqrt(p_up * (1 - p_up) / 1000)
end

@testset "PTT update decisions" begin
    ladder = TrajectoryLadder(BinaryRBM(4, 2); nchains = 10) # α = 0.3, αmin = 0.1
    for (acceptance, since_checkpoint, status) in (
            (0.5, 0, :accepted), (0.2, 2, :frozen), # freeze when acceptance < α
            (0.2, 1, :rejected), # ... but not one update after the last checkpoint
            (0.05, 5, :rejected), # acceptance < αmin
        )
        ladder.acceptance, ladder.since_checkpoint = acceptance, since_checkpoint
        @test RBMs._ptt_status(ladder) === status
    end
end

#= Annealing in a single step from the independent-site model to the bimodal model above
loses overlap, so the step is rejected and refined; intermediate checkpoints are only
possible after such a rejection. =#
@testset "TrajectoryLadder with a coarse anneal" begin
    rbm = RBM(Spin(; θ = fill(0.03, 10)), Gaussian(; θ = zeros(1), γ = ones(1)), fill(0.8, 10, 1))
    ladder = TrajectoryLadder(rbm; nchains = 1000, anneal = 1)
    @test length(ladder.checkpoints) > 2
    @test abs(log_partition(ladder) - log_partition(rbm)) < 0.1
end

@testset "TrajectoryLadder of $(nameof(typeof(model)))" for model in (
        BinaryRBM(randn(8) / 2, randn(4) / 2, 1.5randn(8, 4)),
        center(BinaryRBM(randn(8) / 2, randn(4) / 2, 1.5randn(8, 4)), rand(8), rand(4)),
        standardize(BinaryRBM(randn(8) / 2, randn(4) / 2, 1.5randn(8, 4)), rand(8), rand(4), 0.5 .+ rand(8), 0.5 .+ rand(4)),
    )
    ladder = TrajectoryLadder(model; nchains = 1000)
    @test ladder.rbm === model
    @test iszero(first(ladder.checkpoints).w)
    @test last(ladder.checkpoints).w == model.w
    @test last(ladder.checkpoints).w !== model.w
    @test first(ladder.logZ) ≈ log_partition(first(ladder.checkpoints))
    @test all(isapprox.(ladder.logZ, log_partition.(ladder.checkpoints); atol = 0.05))
    @test log_partition(ladder) ≈ log_partition(model) atol = 0.05
    states = enumerate_states(model.visible)
    @test maximum(abs, log_likelihood(ladder, states) - RBMs.log_likelihood(model, states)) < 0.05
    p = softmax(-free_energy(model, states))
    @test total_variation(empirical_distribution(ladder.chains, states), p) < 4tv_noise(p, 1000)

    @test_throws ArgumentError TrajectoryLadder(model; nchains = 10, nreservoir = 5)
    @test_throws ArgumentError TrajectoryLadder(model; nchains = 10, α = 0.1, αmin = 0.2)
end

@testset "ptt!" begin
    ξ = rand(Bool, 8)
    data = falses(8, 1000) # two noisy modes, with weights 0.7 and 0.3
    for n in 1:1000
        data[:, n] .= (rand() < 0.7 ? ξ : .!ξ) .⊻ (rand(8) .< 0.1)
    end
    rbm = BinaryRBM(8, 4)
    initialize!(rbm, data)
    ladder = TrajectoryLadder(rbm; nchains = 500, α = 0.6) # frequent checkpoints
    K₀ = length(ladder.checkpoints)
    ll₀ = mean(RBMs.log_likelihood(rbm, data))
    nfrozen = Ref(0)
    state, ps = ptt!(
        rbm, data; ladder, batchsize = 100, iters = 1000, optim = CossimDescent(0.02, 0.1),
        callback = (; vm, ladder, _...) -> begin
            @assert vm === ladder.chains
            nfrozen[] = length(ladder.checkpoints)
        end,
    )
    @test nfrozen[] == length(ladder.checkpoints) > K₀
    @test mean(RBMs.log_likelihood(rbm, data)) > ll₀ + 1
    @test log_partition(ladder) ≈ log_partition(rbm) atol = 0.05
    @test all(isapprox.(ladder.logZ, log_partition.(ladder.checkpoints); atol = 0.05))
    states = enumerate_states(rbm.visible)
    p = softmax(-free_energy(rbm, states))
    @test total_variation(empirical_distribution(ladder.chains, states), p) < 6tv_noise(p, 500)

    other = BinaryRBM(8, 4)
    @test_throws ArgumentError ptt!(other, data; ladder, batchsize = 100)
end

@testset "ptt! rejects updates that lose overlap" begin
    data = falses(8, 100)
    data[:, 1:2:end] .= true
    rbm = BinaryRBM(8, 4)
    initialize!(rbm, data)
    ladder = TrajectoryLadder(rbm; nchains = 500)
    state, _ = ptt!(rbm, data; ladder, batchsize = 100, iters = 20, optim = Descent(1000.0))
    @test state.w.rule.eta < 1000 # halved at least once
    @test all(isfinite, rbm.w)
    for _ in 1:20 # let the chains catch up with the last (large) update
        RBMs._ptt_update!(ladder, rbm; steps = 1)
    end
    @test log_partition(ladder) ≈ log_partition(rbm) atol = 0.1
end
