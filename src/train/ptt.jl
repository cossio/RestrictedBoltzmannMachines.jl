#= Parallel Trajectory Tempering (PTT).

N. Béreux, A. Decelle, C. Furtlehner, B. Seoane, "Equilibrium Training of Energy-Based
Models with Parallel Trajectory Tempering", arXiv:2607.27077 (2026).

PTT keeps the persistent chains of the model being trained at equilibrium by replica
exchange with frozen checkpoints of its own training trajectory. A checkpoint is frozen
whenever the swap acceptance between the last checkpoint and the current model drops below
`α`. Equilibrium samples of the last checkpoint are kept in a reservoir, so only the chains
of the current model are simulated: each sweep proposes to exchange every chain with a
random reservoir member, and then runs Gibbs sampling. The ladder starts at the
independent-site model (`w = 0`), which is sampled exactly and whose partition function is
known; partition functions of later checkpoints follow by the Bennett acceptance ratio
between the equilibrium samples of consecutive checkpoints. =#

# chains of a new checkpoint are thermalized for this many autocorrelation times (as in the paper),
const PTT_THERMALIZATION = 20
# and then collected every this many integrated autocorrelation times (as in the paper)
const PTT_DECORRELATION = 2
# consecutive rejected updates after which `ptt!` gives up
const PTT_MAX_REJECTIONS = 30
# learning rate halvings after which `ptt!` warns that training may stall
const PTT_WARN_HALVINGS = 10

"""
    TrajectoryLadder(rbm; nchains, nreservoir = 10nchains, α = 0.3, αmin = 0.1, sweeps = 1, steps = 1, anneal = 100)

Persistent state of Parallel Trajectory Tempering for training `rbm` with [`ptt!`](@ref):
frozen checkpoints along the training trajectory of `rbm`, their log-partition functions,
a reservoir of `nreservoir` equilibrium samples of the last checkpoint, and `nchains`
persistent chains of `rbm`. The algorithm is described in
[Equilibrium training with `ptt!`](@ref ptt_training).

Every update runs `sweeps` sweeps, each exchanging the chains with reservoir samples and
then running Gibbs sampling. `α` is the swap acceptance between the last checkpoint and
`rbm` below which a new checkpoint is frozen, and `αmin` the one below which the update is
rejected.

`rbm` is an `RBM` or a `StandardizedRBM` (including a `CenteredRBM`), freshly initialized
by [`initialize!`](@ref). The ladder starts at the independent-site model obtained by
setting the weights of `rbm` to zero, whose partition function is known, and is extended
along `anneal` steps scaling the weights up to those of `rbm`, which becomes the last
checkpoint. `steps` are the Gibbs steps per sweep used meanwhile.

The checkpoints and their log-partition functions are kept in `ladder.checkpoints` and
`ladder.logZ`, the persistent chains in `ladder.chains`, equilibrium samples of the last
checkpoint in `ladder.samples`, and the last swap acceptance in `ladder.acceptance`. As
diagnostics of the thermalization of the last checkpoint, `ladder.τint` and `ladder.τexp`
hold the integrated and exponential autocorrelation times, in sweeps, of the ladder level
of its chains. See also [`log_partition(ladder)`](@ref log_partition(::TrajectoryLadder)),
[`log_likelihood(ladder, v)`](@ref log_likelihood(::TrajectoryLadder, ::AbstractArray)), and
[`freeze!`](@ref), which freezes `rbm` as a checkpoint to sample it at equilibrium.
"""
Base.@kwdef mutable struct TrajectoryLadder{M, A <: AbstractArray}
    const rbm::M # model being trained (not a copy)
    const checkpoints::Vector{M} # frozen copies of `rbm` along its training trajectory
    const logZ::Vector{Float64} # log-partition functions of the checkpoints
    const chains::A # persistent chains of `rbm`
    const chains_F::Vector{Float64} # their free energies under the model they were last sampled from
    const samples::A # equilibrium samples of the last checkpoint
    const samples_F::Vector{Float64} # their free energies under the last checkpoint
    const reservoir::A = copy(samples) # working copy of `samples`, exchanged with `chains`
    const reservoir_F::Vector{Float64} = copy(samples_F)
    acceptance::Float64 = 1.0 # last swap acceptance between the last checkpoint and `rbm`
    since_checkpoint::Int = 0 # updates since the last checkpoint was frozen or restored
    rejections::Int = 0 # consecutive rejected updates
    τint::Float64 = NaN # integrated and exponential autocorrelation times (in sweeps) of the
    τexp::Float64 = NaN # chains' ladder level, measured when building the last reservoir
    const sweeps::Int
    const α::Float64
    const αmin::Float64
end

function TrajectoryLadder(
        rbm; nchains::Int, nreservoir::Int = 10nchains, α::Real = 0.3, αmin::Real = 0.1,
        sweeps::Int = 1, steps::Int = 1, anneal::Int = 100
    )
    nchains > 0 || throw(ArgumentError("nchains must be positive"))
    nreservoir ≥ nchains || throw(ArgumentError("nreservoir must be at least nchains"))
    0 ≤ αmin < α ≤ 1 || throw(ArgumentError("expected 0 ≤ αmin < α ≤ 1"))
    sweeps > 0 && anneal > 0 || throw(ArgumentError("sweeps and anneal must be positive"))
    independent = deepcopy(rbm)
    independent.w .= 0
    samples = _default_fantasy_chains(independent, nreservoir) # exact samples
    samples_F = _free_energies(independent, samples)
    chains = _default_fantasy_chains(independent, nchains)
    ladder = TrajectoryLadder(;
        rbm, checkpoints = [independent], logZ = [Float64(log_partition_zero_weight(independent))],
        chains, chains_F = _free_energies(independent, chains), samples, samples_F, sweeps, α, αmin
    )
    _anneal!(ladder; steps, nsteps = anneal)
    return ladder
end

"""
    log_partition(ladder::TrajectoryLadder)

Estimate of the log-partition function of the model trained with `ladder`, by the Bennett
acceptance ratio between equilibrium samples of the last checkpoint and the chains of the
model. The chains are reweighted from the model they were last sampled from (before the
last parameter update) to the current one.
"""
function log_partition(ladder::TrajectoryLadder)
    (; rbm, chains, chains_F, samples, samples_F) = ladder
    F = _free_energies(rbm, chains)
    W₀ = _free_energies(rbm, samples) - samples_F # work from the last checkpoint to `rbm`
    W₁ = _free_energies(last(ladder.checkpoints), chains) - F # and back
    return last(ladder.logZ) + _bennett(W₀, W₁; logw = chains_F - F)
end

"""
    log_likelihood(ladder::TrajectoryLadder, v)

Log-likelihood of `v` under the model trained with `ladder`, using the partition function
estimate [`log_partition(ladder)`](@ref log_partition(::TrajectoryLadder)).
"""
log_likelihood(ladder::TrajectoryLadder, v::AbstractArray) =
    -free_energy(ladder.rbm, v) .- log_partition(ladder)

"""
    ptt!(rbm, data; kwargs...)

Train an `RBM` or a `StandardizedRBM` (including a `CenteredRBM`) with Parallel Trajectory
Tempering (PTT; Béreux, Decelle, Furtlehner, Seoane, arXiv:2607.27077).

Like [`pcd!`](@ref), but the persistent chains are kept at equilibrium by replica exchange
with frozen checkpoints of the training trajectory, held by a [`TrajectoryLadder`](@ref).
Each update runs `ladder.sweeps` sweeps, exchanging the chains with equilibrium samples of
the last checkpoint and then running `steps` Gibbs steps, before the gradient step.

An update that loses overlap with the last checkpoint is rejected: `rbm` is restored to that
checkpoint, the learning rate of the optimiser is halved for good, and the optimiser forgets
its past gradients (momenta, moment estimates). `ptt!` warns once it has halved the
learning rate $PTT_WARN_HALVINGS times, and throws an error if more than
$PTT_MAX_REJECTIONS updates in a row are rejected. See
[Equilibrium training with `ptt!`](@ref ptt_training) for when updates are rejected, and
what to do if rejections stall training.

`data` must have shape `(size(rbm.visible)..., nsamples)`. As in [`pcd!`](@ref) and the
reference implementation, every minibatch is drawn at random from `data`, independently of
the others. The offsets and scales of a `StandardizedRBM` are set and updated as by
[`pcd!`](@ref), which leaves its distribution unchanged; a plain `RBM` is trained as the
equivalent `StandardizedRBM` whose offsets and scales are fixed to zero and one.

# Keyword arguments
- `ladder`: the [`TrajectoryLadder`](@ref) of `rbm`, by default a new one with
  `nchains = min(batchsize, nsamples)` chains. To continue training, pass the ladder of the
  previous run.
- `steps::Int=1`: Gibbs steps per sweep.
- `optim::AbstractRule=Adam(1e-4)`: optimizer rule from `Optimisers.jl`, with a learning
  rate `eta`. [`CossimDescent`](@ref) is the optimiser used in the paper.
- `callback=Returns(nothing)`: called after every update as
  `callback(; rbm, optim, state, ps, iter, vd, wd, ∂, vm, ladder)`, where `vm` are the
  chains of the ladder. Slurp unused keywords with a trailing `_...`.
- `batchsize`, `iters`, `wts`, `moments`, `damping`, `ϵv`, `ϵh`,
  `regularization`, `zerosum`, `rescale`, `ps`, `state`: as for [`pcd!`](@ref).

Returns `(state, ps)`. The ladder then lags behind `rbm`: `ladder.samples` are equilibrium
samples of the last checkpoint, and `ladder.chains` sample the model before its last update.
[`freeze!`](@ref) freezes the trained `rbm` as a checkpoint, to sample it at equilibrium.
"""
function ptt!(rbm::RBM, data::AbstractArray; callback = Returns(nothing), kwargs...)
    std_rbm = PlainStandardizedRBM(rbm) # shares the layers and weights of `rbm`
    # the callback receives `rbm` rather than `std_rbm` (the rightmost keyword wins)
    return ptt!(std_rbm, data; callback = (; kw...) -> callback(; kw..., rbm), kwargs...)
end

function ptt!(
        rbm::StandardizedRBM,
        data::AbstractArray;
        batchsize::Int = 1,
        iters::Int = 1, # number of gradient updates
        wts::AbstractVector{<:Real} = uniform_wts(rbm.visible, data), # data weights
        steps::Int = 1, # Gibbs steps per sweep
        ladder::TrajectoryLadder = TrajectoryLadder(rbm; nchains = min(batchsize, size(data)[end]), steps),
        optim::AbstractRule = Adam(1.0e-4), # optimizer rule
        moments = moments_from_samples(rbm.visible, data; wts), # sufficient statistics for visible layer
        damping::Real = 1 // 100, # of the hidden standardization updates
        ϵv::Real = 0, ϵh::Real = 0, # pseudocounts for the visible and hidden variances
        regularization::AbstractRegularizer = CompositeRegularizer(),

        # gauge
        zerosum::Bool = true, # zerosum gauge for Potts layers
        rescale::Bool = true, # fix the scale gauge of the hidden units

        callback = Returns(nothing), # called for every batch

        # parameters to optimize
        ps = (; visible = rbm.visible.par, hidden = rbm.hidden.par, w = rbm.w),
        state = setup(optim, ps),
    )
    # the ladder samples `rbm`, or the plain `RBM` whose layers and weights `rbm` shares
    model = ladder.rbm
    model.visible === rbm.visible && model.hidden === rbm.hidden && model.w === rbm.w ||
        throw(ArgumentError("the ladder was built for another model"))
    hasproperty(optim, :eta) || throw(ArgumentError("ptt! halves the learning rate `eta` of the optimiser, which $optim lacks"))
    @assert 0 ≤ damping ≤ 1
    wts_mean, batchsize = _pcd_check_args(rbm, data, wts, batchsize)

    standardize_visible_from_data!(rbm, data; wts, ϵ = ϵv)
    standardize_hidden_from_v!(rbm, data; wts, ϵ = ϵh)
    # zerosum! first because rescaling preserves the zero-sum gauge
    zerosum && zerosum!(rbm)
    rescale && rescale_hidden_activations!(rbm)

    halvings = 0 # of the learning rate, by rejected updates
    for iter in 1:iters
        idx = sample(1:_nsamples(data), batchsize; replace = false)
        vd, wd = data[.., idx], wts[idx]

        # negative phase first, since a rejected update restores the parameters
        if _ptt_update!(ladder, model; steps) === :rejected
            ladder.rejections ≤ PTT_MAX_REJECTIONS ||
                error("PTT lost equilibrium after $PTT_MAX_REJECTIONS consecutive learning rate halvings")
            _restart_optimiser!(state, ps)
            halvings += 1
            halvings == PTT_WARN_HALVINGS && @warn """
            PTT rejected $halvings updates, so the learning rate is down to 1/$(2^halvings) of its \
            initial value; training may stall. An optimiser without momentum, such as Descent, may \
            train better: see "Equilibrium training with ptt!" in the documentation."""
        end
        ∂m = ∂free_energy(rbm, ladder.chains)

        # positive phase
        ∂d = ∂free_energy(rbm, vd; wts = wd, moments)

        # weighted minibatch bias correction, in the gradient eltype
        batch_weight = convert(float(real(eltype(∂d.w))), mean(wd) / wts_mean)
        ∂ = (∂d - ∂m) * batch_weight

        # regularization, and projection of the gradient onto the zerosum gauge
        ∂regularize!(∂, rbm, regularization)
        zerosum && zerosum!(∂, rbm)

        # feed gradient to Optimiser rule
        gs = (; visible = ∂.visible, hidden = ∂.hidden, w = ∂.w)
        state, ps = update!(state, ps, gs)
        _validate_layer_parameters(rbm)

        # these leave the distribution unchanged
        standardize_hidden_from_v!(rbm, vd; wts = wd, damping, ϵ = ϵh)
        zerosum && zerosum!(rbm)
        rescale && rescale_hidden_activations!(rbm)

        callback(; rbm, optim, state, ps, iter, vd, wd, ∂, vm = ladder.chains, ladder)
    end
    return state, ps
end

"""
    freeze!(ladder::TrajectoryLadder; steps = 1)

Freeze the model trained with `ladder` as a new checkpoint, to sample it at equilibrium:
after [`ptt!`](@ref), the ladder only holds equilibrium samples of the last checkpoint, which
the model moved away from. As for the checkpoints frozen by `ptt!`, the chains of the model
are thermalized and collected into a new reservoir, with `steps` Gibbs steps per sweep, and
its log-partition function is appended to `ladder.logZ`.

Returns `true` if the model was frozen: `ladder.samples` are then equilibrium samples of the
model, and `last(ladder.logZ)`, `ladder.τint` and `ladder.τexp` refer to it. Returns `false`
if the model lost overlap with the last checkpoint (their swap acceptance is below
`ladder.αmin`, before or after thermalizing the chains): the model is then restored to the
last checkpoint, and the chains to its equilibrium samples, as at a rejected update of
`ptt!`.
"""
freeze!(ladder::TrajectoryLadder; steps::Int = 1) =
    _ptt_update!(ladder, ladder.rbm; steps, freeze = true) === :frozen

#= Restarts the optimiser after a rejected update: halves the learning rate of every
parameter array in the optimiser state tree of the parameters `ps`, and discards the memory
of past gradients (momenta, moment estimates), which would otherwise repeat the rejected
step. =#
_restart_optimiser!(tree::Union{Tuple, NamedTuple}, ps) = foreach(_restart_optimiser!, tree, ps)
_restart_optimiser!(::Tuple{}, ps) = nothing # parameters without optimiser state
# `leaf` is the `Optimisers.Leaf` of the parameters `x`
_restart_optimiser!(leaf, x) = ((leaf.rule, leaf.state) = _restart_optimiser(leaf.rule, leaf.state, x); nothing)
_restart_optimiser(o::CossimDescent, (g, η), x::AbstractArray) = o, (zero(g), η / 2)
function _restart_optimiser(o::AbstractRule, state, x::AbstractArray)
    o = Optimisers.adjust(o, o.eta / 2)
    return o, Optimisers.init(o, x)
end

#= One PTT update of the chains of `model`, which moved along its trajectory since the last
update. Returns `:rejected` if `model` lost overlap with the last checkpoint (as decided by
`_ptt_status` and `_push_checkpoint!`), in which case `model` is restored to that
checkpoint, and the reservoir and the chains to its equilibrium samples; `:frozen` if
`model` was frozen as a new checkpoint; and `:accepted` otherwise. With `freeze`, `model`
is frozen unless rejected.

The chains still sample the model before its last move, so the acceptance that decides
between these outcomes reweights them to `model`. Otherwise, the acceptance of a large step
is overestimated until the chains catch up. =#
function _ptt_update!(ladder::TrajectoryLadder, model; steps::Int, freeze::Bool = false)
    proposal = _propose_exchange(ladder, model)
    w = softmax(ladder.chains_F - proposal.Fθx) # reweights the chains to `model`
    ladder.acceptance = dot(w, min.(1, exp.(proposal.Δ)))
    status = _ptt_status(ladder; freeze)
    status === :rejected && return _reject!(ladder, model)
    _exchange!(ladder, proposal)
    if status === :frozen
        _push_checkpoint!(ladder, model; steps) || return _reject!(ladder, model)
    end
    ladder.chains .= sample_v_from_v(model, ladder.chains; steps)
    for _ in 2:ladder.sweeps
        _sweep!(ladder, model; steps)
    end
    ladder.chains_F .= _free_energies(model, ladder.chains)
    ladder.since_checkpoint += 1
    ladder.rejections = 0
    return status
end

#= Outcome of an update, from the acceptance just measured: `:accepted` while it stays at
least `α`, `:frozen` once it falls below `α` but not below `αmin`, and `:rejected` otherwise.
An acceptance below `α` within two updates of the last checkpoint rejects the update instead
of freezing a checkpoint every other update, so that the learning rate is halved. With
`freeze`, `model` is frozen whenever the acceptance is at least `αmin`. =#
function _ptt_status(ladder::TrajectoryLadder; freeze::Bool = false)
    ladder.acceptance ≥ ladder.αmin || return :rejected
    freeze && return :frozen
    ladder.acceptance ≥ ladder.α && return :accepted
    return ladder.since_checkpoint > 1 ? :frozen : :rejected
end

# exchanges the chains with the reservoir, then runs Gibbs sampling; returns the accepted swaps
function _sweep!(ladder::TrajectoryLadder, model; steps::Int)
    accept = _exchange!(ladder, _propose_exchange(ladder, model))
    ladder.chains .= sample_v_from_v(model, ladder.chains; steps)
    return accept
end

# resets the working reservoir to the equilibrium samples of the last checkpoint
function _reset_reservoir!(ladder::TrajectoryLadder)
    ladder.reservoir .= ladder.samples
    ladder.reservoir_F .= ladder.samples_F
    return ladder
end

# restores `model` to the last checkpoint, and the reservoir and the chains to its samples
function _reject!(ladder::TrajectoryLadder, model)
    _copyto_model!(model, last(ladder.checkpoints))
    _reset_reservoir!(ladder)
    idx = sample(1:_nsamples(ladder.samples), _nsamples(ladder.chains); replace = false)
    ladder.chains .= ladder.samples[.., idx]
    ladder.chains_F .= ladder.samples_F[idx]
    ladder.since_checkpoint = 0
    ladder.rejections += 1
    return :rejected
end

#= Proposes to exchange each chain of `model` with a random reservoir member, an equilibrium
sample of the last checkpoint. `Δ` holds the log Metropolis ratios. =#
function _propose_exchange(ladder::TrajectoryLadder, model)
    x = ladder.chains
    idx = sample(1:_nsamples(ladder.reservoir), _nsamples(x); replace = false)
    y = ladder.reservoir[.., idx]
    Fy = ladder.reservoir_F[idx]
    Fx = _free_energies(last(ladder.checkpoints), x)
    Fθx = _free_energies(model, x)
    Δ = (Fθx - Fx) - (_free_energies(model, y) - Fy)
    return (; idx, y, Fx, Fθx, Δ)
end

# applies the exchange `proposal` by the Metropolis rule; returns the accepted swaps
function _exchange!(ladder::TrajectoryLadder, proposal::NamedTuple)
    (; idx, y, Fx, Δ) = proposal
    accept = log.(rand(length(Δ))) .< Δ
    swapped = findall(accept)
    ladder.reservoir[.., idx[swapped]] = ladder.chains[.., swapped]
    ladder.reservoir_F[idx[swapped]] = Fx[swapped]
    ladder.chains[.., swapped] = y[.., swapped]
    return accept
end

#= Freezes a copy of `model` as a new checkpoint. Its chains are thermalized by exchanges
with the reservoir of the previous checkpoint, for `PTT_THERMALIZATION` times the
autocorrelation time of their ladder level (and at least `minsweeps` sweeps), and then
collected every `PTT_DECORRELATION` integrated autocorrelation times into new equilibrium
samples, as in the paper.

Returns `false`, with the checkpoints unchanged, if the chains fail to thermalize within
`maxsweeps` sweeps, or if their swap acceptance falls below `αmin`. The online acceptance
then overestimated the overlap of `model` with the last checkpoint, because its chains
lagged behind `model`. The swap acceptance is measured over the second half of the run. It
typically decreases during thermalization, from its value for chains fed by the reservoir
to its equilibrium value, so the run stops as soon as it falls below `αmin`. The reference
implementation also checks both conditions.

The working reservoir is first reset to the equilibrium samples of the previous checkpoint,
because exchanges write the chains back into it, which biases it in two ways: chains lagging
behind the moving model are not at equilibrium, and, when Gibbs sampling cannot move between
modes, the swaps conserve the number of configurations of each mode in chains and reservoir
together, so that a shift of the model's mode weights is partly absorbed by the reservoir. =#
function _push_checkpoint!(ladder::TrajectoryLadder, model; steps::Int, minsweeps::Int = 20, maxsweeps::Int = 10_000)
    checkpoint = deepcopy(model)
    _reset_reservoir!(ladder)
    sweeps(n) = stack(_sweep!(ladder, checkpoint; steps) for _ in 1:n) # accepted swaps, chains × sweeps

    swaps = sweeps(minsweeps)
    τint, τexp = _autocorrelation_times(swaps)
    while _late_mean(swaps) ≥ ladder.αmin && size(swaps, 2) < PTT_THERMALIZATION * max(τint, τexp)
        size(swaps, 2) < maxsweeps || return false
        swaps = hcat(swaps, sweeps(size(swaps, 2))) # doubles the run
        τint, τexp = _autocorrelation_times(swaps)
    end
    _late_mean(swaps) ≥ ladder.αmin || return false

    samples = similar(ladder.samples)
    for block in Iterators.partition(1:_nsamples(samples), _nsamples(ladder.chains))
        for _ in 1:ceil(Int, PTT_DECORRELATION * τint)
            _sweep!(ladder, checkpoint; steps)
        end
        samples[.., block] = ladder.chains[.., 1:length(block)]
    end

    samples_F = _free_energies(checkpoint, samples)
    W₀ = _free_energies(checkpoint, ladder.samples) - ladder.samples_F # work from the last checkpoint to the new one
    W₁ = _free_energies(last(ladder.checkpoints), samples) - samples_F # and back
    push!(ladder.logZ, last(ladder.logZ) + _bennett(W₀, W₁))
    push!(ladder.checkpoints, checkpoint)
    ladder.samples .= samples
    ladder.samples_F .= samples_F
    _reset_reservoir!(ladder)
    ladder.chains_F .= _free_energies(checkpoint, ladder.chains)
    ladder.τint, ladder.τexp = τint, τexp
    ladder.since_checkpoint = 0
    return true
end

# mean over the second half of the sweeps (columns) of `swaps`
_late_mean(swaps::AbstractMatrix) = mean(view(swaps, :, (size(swaps, 2) ÷ 2 + 1):size(swaps, 2)))

#= Builds the initial ladder, from the independent-site model to `ladder.rbm`, along models
whose weights are those of `ladder.rbm` scaled by k / `nsteps`, for k = 1, …, `nsteps`.
`ladder.rbm` is frozen as the last checkpoint, so that rejected training updates never move
back further than the initial model. =#
function _anneal!(ladder::TrajectoryLadder; steps::Int, nsteps::Int)
    model = deepcopy(ladder.rbm)
    for k in 1:nsteps
        model.w .= (k / nsteps) .* ladder.rbm.w
        _ptt_update!(ladder, model; steps, freeze = k == nsteps) === :rejected &&
            error("PTT failed to anneal from the independent-site model to the initial model; increase `anneal`")
    end
    return ladder
end

#= Integrated and exponential autocorrelation times (in sweeps) of the ladder level of each
chain, whose diffusion the paper uses to measure thermalization (after Alvarez Baños et al.,
J. Stat. Mech. (2010) P06026). With the reservoir in place of the previous checkpoint, the
level of a chain flips at each of its accepted exchanges, given by `swaps` (chains × sweeps).
The integrated time uses Sokal's self-consistent window, and the exponential time is the
slowest decay of the autocorrelation while it is above 0.05. =#
function _autocorrelation_times(swaps::AbstractMatrix{Bool})
    s = @. ifelse(isodd($cumsum(swaps; dims = 2)), -1.0, 1.0) # level of each chain, as ±1
    T = size(s, 2)
    τint = 0.5
    τexp = 0.0
    for t in 1:(T ÷ 2)
        C = dot(view(s, :, (1 + t):T), view(s, :, 1:(T - t))) / (size(s, 1) * (T - t))
        C > 0.05 && (τexp = max(τexp, -t / log(C)))
        τint += C
        t ≥ 6τint && break
    end
    return max(τint, 0.5), τexp
end

#= Bennett acceptance ratio estimate of log(Z₁ / Z₀) from the works `W₀ = F₁ - F₀` of
equilibrium samples of model 0 and `W₁ = F₀ - F₁` of samples of model 1 (Bennett,
J. Comput. Phys. 22, 245 (1976); Shirts et al., Phys. Rev. Lett. 91, 140601 (2003)). The
samples of model 1 can carry importance log-weights `logw`, entering through their
normalized weights and effective number. Solves the self-consistent equation for
Δf = log(Z₀ / Z₁) by bisection. =#
function _bennett(W₀::AbstractVector, W₁::AbstractVector; logw = Zeros(length(W₁)))
    p₁ = softmax(Array{Float64}(logw))
    n₁ = 1 / sum(abs2, p₁) # effective number of samples x₁
    M = log(length(W₀) / n₁)
    g(Δf) = sum(w -> logistic(Δf - M - w), W₀) - n₁ * sum(p₁ .* logistic.(M .- W₁ .- Δf))
    lo = hi = -logmeanexp(-W₀) # one-sided (exponential averaging) estimate
    while g(lo) > 0
        lo -= 1 + abs(lo)
    end
    while g(hi) < 0
        hi += 1 + abs(hi)
    end
    for _ in 1:100
        mid = (lo + hi) / 2
        lo < mid < hi || break
        g(mid) < 0 ? (lo = mid) : (hi = mid)
    end
    return -(lo + hi) / 2
end

_nsamples(x::AbstractArray) = size(x, ndims(x))

# free energies of the samples `x` under `model`, on the host in double precision
_free_energies(model, x::AbstractArray) = convert(Vector{Float64}, Array(free_energy(model, x)))

# copies the parameters of `src` into those of `dst`, a model of the same type
_copyto_model!(dst::AbstractArray, src::AbstractArray) = copyto!(dst, src)
function _copyto_model!(dst::T, src::T) where {T}
    foreach(f -> _copyto_model!(getfield(dst, f), getfield(src, f)), fieldnames(T))
    return dst
end
