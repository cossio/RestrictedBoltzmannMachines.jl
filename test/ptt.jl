import Random
import RestrictedBoltzmannMachines as RBMs
using Test: @test, @testset, @test_logs, @test_throws
using Statistics: mean
using LogExpFunctions: softmax
using StatsBase: sample, Weights
using EllipsisNotation: (..)
using Optimisers: Adam, ClipGrad, Descent, Nesterov, setup, update!
using RestrictedBoltzmannMachines: RBM, BinaryRBM, Binary, Spin, Potts, Gaussian,
    TrajectoryLadder, CossimDescent, ptt!, initialize!, free_energy,
    log_partition, log_likelihood, collect_states, standardize, StandardizedRBM,
    unstandardize, delta_energy

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

# a ladder with a single checkpoint, `base`, without freezing or rejecting updates, which run `sweeps` sweeps
function fixed_ladder(model, base, states; nchains::Int, nreservoir::Int, sweeps::Int)
    chains = exact_samples(states, softmax(-free_energy(model, states)), nchains)
    samples = exact_samples(states, softmax(-free_energy(base, states)), nreservoir)
    return TrajectoryLadder(;
        rbm = model, checkpoints = [base], logZ = [0.0], chains, chains_F = free_energy(model, chains),
        samples, samples_F = free_energy(base, samples), sweeps, α = 0.0, αmin = 0.0
    )
end

# `n` samples of two noisy modes, a random pattern and its complement, with weights 0.7 and 0.3
function two_modes(N::Int, n::Int)
    ξ = rand(Bool, N)
    data = falses(N, n)
    for s in 1:n
        data[:, s] .= (rand() < 0.7 ? ξ : .!ξ) .⊻ (rand(N) .< 0.1)
    end
    return data
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
end

# after a rejected update, the optimiser restarts with half the learning rate
@testset "PTT restarts the optimiser" begin
    ps = (; a = [1.0, -1.0], b = [2.0])
    gs = (; a = [1.0, 0.5], b = [-1.0])
    for (optim, halved) in (
            (CossimDescent(0.1, 0.2), CossimDescent(0.05, 0.2)), (Adam(0.1), Adam(0.05)),
            (Nesterov(0.1, 0.9), Nesterov(0.05, 0.9)), (Descent(0.1), Descent(0.05)),
        )
        state = setup(optim, ps)
        update!(state, deepcopy(ps), gs) # builds up momenta
        RBMs._restart_optimiser!(state, ps)
        @test update!(state, deepcopy(ps), gs)[2] == update!(setup(halved, ps), deepcopy(ps), gs)[2]
    end
    ps = (; a = [1.0], b = 1) # the non-trainable `b` has an empty state
    state = setup(Adam(0.1), ps)
    RBMs._restart_optimiser!(state, ps)
    @test state.a.rule.eta == 0.05
end

#= Replica exchange with the reservoir, followed by Gibbs sampling, must leave the joint
distribution of the chains and the reservoir invariant: starting both exactly at
equilibrium, they must stay at equilibrium within Monte-Carlo error, through updates of
two sweeps each. =#
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
    ladder = fixed_ladder(model, base, states; nchains = 20_000, nreservoir = 40_000, sweeps = 2)
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
    for (acceptance, since_checkpoint, status) in ((0.5, 0, :frozen), (0.2, 1, :rejected))
        ladder.acceptance, ladder.since_checkpoint = acceptance, since_checkpoint
        @test RBMs._ptt_status(ladder; freeze = true) === status
    end
end

@testset "TrajectoryLadder of $name" for (name, model) in (
        ("RBM", BinaryRBM(randn(8) / 2, randn(4) / 2, 1.5randn(8, 4))),
        ("CenteredRBM", standardize(BinaryRBM(randn(8) / 2, randn(4) / 2, 1.5randn(8, 4)), rand(8), rand(4))),
        ("StandardizedRBM", standardize(BinaryRBM(randn(8) / 2, randn(4) / 2, 1.5randn(8, 4)), rand(8), rand(4), 0.5 .+ rand(8), 0.5 .+ rand(4))),
    )
    ladder = TrajectoryLadder(model; nchains = 1000)
    @test ladder.rbm === model
    @test iszero(first(ladder.checkpoints).w)
    @test last(ladder.checkpoints).w == model.w
    @test last(ladder.checkpoints).w !== model.w
    @test first(ladder.logZ) ≈ log_partition(first(ladder.checkpoints))
    @test all(isapprox.(ladder.logZ, log_partition.(ladder.checkpoints); atol = 0.05))
    @test log_partition(ladder) ≈ log_partition(model) atol = 0.05
    # the estimate refers to the parametrization of `model`
    @test log_partition(unstandardize(model)) ≈ log_partition(ladder) + delta_energy(model) atol = 0.05
    states = enumerate_states(model.visible)
    @test maximum(abs, log_likelihood(ladder, states) - RBMs.log_likelihood(model, states)) < 0.05
    p = softmax(-free_energy(model, states))
    @test total_variation(empirical_distribution(ladder.chains, states), p) < 4tv_noise(p, 1000)

    @test_throws ArgumentError TrajectoryLadder(model; nchains = 10, nreservoir = 5)
    @test_throws ArgumentError TrajectoryLadder(model; nchains = 10, α = 0.1, αmin = 0.2)
end

# a single anneal step to the bimodal model above loses overlap
@testset "TrajectoryLadder with a coarse anneal" begin
    rbm = RBM(Spin(; θ = fill(0.03, 10)), Gaussian(; θ = zeros(1), γ = ones(1)), fill(0.8, 10, 1))
    @test_throws ErrorException TrajectoryLadder(rbm; nchains = 1000, anneal = 1)
end

#= Bennett's acceptance ratio is exact for models whose energies differ by a constant, c,
whatever the importance weights, and accurate for unit Gaussians centered at 0 and μ, the
second with energy shifted by c. =#
@testset "Bennett acceptance ratio" begin
    @test RBMs._bennett(fill(0.7, 100), fill(-0.7, 30)) ≈ -0.7
    @test RBMs._bennett(fill(0.7, 100), fill(-0.7, 30); logw = randn(30)) ≈ -0.7
    μ, c = 1.0, 0.4
    x₀, x₁ = randn(100_000), μ .+ randn(100_000)
    W₀ = @. (x₀ - μ)^2 / 2 + c - x₀^2 / 2
    W₁ = @. x₁^2 / 2 - (x₁ - μ)^2 / 2 - c
    @test RBMs._bennett(W₀, W₁) ≈ -c atol = 0.02
end

@testset "ptt! of $name" for (name, rbm) in (
        ("RBM", BinaryRBM(8, 4)),
        ("CenteredRBM", standardize(BinaryRBM(8, 4), zeros(8), zeros(4))),
        ("StandardizedRBM", StandardizedRBM(BinaryRBM(8, 4))),
    )
    data = two_modes(8, 1000)
    initialize!(rbm, data)
    ladder = TrajectoryLadder(rbm; nchains = 500, α = 0.8) # frequent checkpoints
    K₀ = length(ladder.checkpoints)
    ll₀ = mean(RBMs.log_likelihood(rbm, data))
    nfrozen = Ref(0)
    state, ps = ptt!(
        rbm, data; ladder, batchsize = 100, iters = 1000, optim = CossimDescent(0.02, 0.1),
        callback = (; vm, ladder, kw...) -> begin
            @assert kw[:rbm] === rbm && vm === ladder.chains
            nfrozen[] = length(ladder.checkpoints)
        end,
    )
    @test nfrozen[] == length(ladder.checkpoints) > K₀
    @test mean(RBMs.log_likelihood(rbm, data)) > ll₀ + 0.5
    @test log_partition(ladder) ≈ log_partition(rbm) atol = 0.05
    @test all(isapprox.(ladder.logZ, log_partition.(ladder.checkpoints); atol = 0.05))
    # each estimate refers to the parametrization of its checkpoint, whose offsets and
    # scales moved along training
    @test log_partition(unstandardize(rbm)) ≈ log_partition(ladder) + delta_energy(rbm) atol = 0.05
    logZ_plain = log_partition.(unstandardize.(ladder.checkpoints))
    @test all(isapprox.(ladder.logZ + delta_energy.(ladder.checkpoints), logZ_plain; atol = 0.05))
    states = enumerate_states(rbm.visible)
    p = softmax(-free_energy(rbm, states))
    @test total_variation(empirical_distribution(ladder.chains, states), p) < 6tv_noise(p, 500)

    other = BinaryRBM(8, 4)
    @test_throws ArgumentError ptt!(other, data; ladder, batchsize = 100)
    @test_throws ArgumentError ptt!(rbm, data; ladder, batchsize = 100, optim = ClipGrad()) # no learning rate
end

#= The chains lag behind `model` at the independent-site model `base`, so that the online
acceptance (≈ 0.61) overestimates the overlap of the two models at equilibrium (≈ 0.054). =#
@testset "PTT rejects checkpoints that do not equilibrate" begin
    base = BinaryRBM(zeros(10), zeros(2), zeros(10, 2))
    ladder = TrajectoryLadder(base; nchains = 1000)
    K = length(ladder.checkpoints)
    model = BinaryRBM(fill(2.0, 10), zeros(2), zeros(10, 2))
    ladder.chains_F .= free_energy(model, ladder.chains) # as if they sampled `model`
    @test RBMs._ptt_update!(ladder, model; steps = 1, freeze = true) === :rejected
    @test ladder.acceptance > 0.5
    @test length(ladder.checkpoints) == K
    @test iszero(model.visible.θ) # restored to the last checkpoint
    @test ladder.reservoir == ladder.samples
    # with enough overlap (≈ 0.13), but more sweeps needed than allowed
    model.visible.θ .= 1.5
    @test !RBMs._push_checkpoint!(ladder, model; steps = 1, maxsweeps = 20)
    @test length(ladder.checkpoints) == K
    @test RBMs._push_checkpoint!(ladder, model; steps = 1)
    @test length(ladder.checkpoints) == K + 1
    @test last(ladder.logZ) ≈ log_partition(model) atol = 0.05
end

@testset "ptt! halves the learning rate at rejections" begin
    data = two_modes(8, 1000)
    rbm = BinaryRBM(8, 4)
    initialize!(rbm, data)
    ladder = TrajectoryLadder(rbm; nchains = 500)
    η₀ = 2.0 # too large: rejected updates, in long runs without the optimiser reset
    η, rejected = [η₀], [false]
    ptt!(
        rbm, data; ladder, batchsize = 100, iters = 300, optim = Nesterov(η₀, 0.9),
        callback = (; state, ladder, _...) -> begin
            push!(η, state.w.rule.eta)
            push!(rejected, ladder.rejections > 0)
        end,
    )
    @test any(rejected)
    # each rejection halves the learning rate, for good
    @test all(t -> η[t] == (rejected[t] ? η[t - 1] / 2 : η[t - 1]), 2:length(η))
    # with momentum, training can stall after many halvings (see ptt!), but the checkpoints
    # stay consistent (with steps this large, the chains can lag behind)
    @test all(isapprox.(ladder.logZ, log_partition.(ladder.checkpoints); atol = 0.2))

    # a learning rate this large takes over 10 halvings, which may stall training
    rbm = BinaryRBM(8, 4)
    initialize!(rbm, data)
    ladder = TrajectoryLadder(rbm; nchains = 500)
    @test_logs (:warn, r"rejected 10 updates") match_mode = :any ptt!(
        rbm, data; ladder, batchsize = 100, iters = 100, optim = Descent(1.0e4)
    )
end
