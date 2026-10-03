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
known, and partition functions of later checkpoints follow by reweighting the reservoirs. =#

"""
    TrajectoryLadder(rbm; nchains, nreservoir = 10nchains, α = 0.3, αmin = 0.1, sweeps = 1, steps = 1, anneal = 100)

Persistent state of Parallel Trajectory Tempering for training `rbm` (see [`ptt!`](@ref)):
frozen checkpoints along the training trajectory of `rbm`, their log-partition functions,
a reservoir of `nreservoir` equilibrium samples of the last checkpoint, and `nchains`
persistent chains of `rbm`.

Every update runs `sweeps` sweeps, each exchanging the chains with reservoir samples and
then running Gibbs sampling. A new checkpoint is frozen when the swap acceptance between
the last checkpoint and `rbm` falls below `α`. If it falls below `αmin`, or below `α` one
update after a checkpoint was frozen, the update is rejected: `rbm` is restored to the last
checkpoint and the learning rate is halved.

The ladder starts at the independent-site model obtained by setting the weights of `rbm`
to zero, and is extended along `anneal` steps scaling the weights up to those of `rbm`,
which becomes the last checkpoint. `steps` are the Gibbs steps per sweep used meanwhile.

The checkpoints and their log-partition functions are kept in `ladder.checkpoints` and
`ladder.logZ`, the persistent chains in `ladder.chains`, equilibrium samples of the last
checkpoint in `ladder.samples`, and the last swap acceptance in `ladder.acceptance`. See
also [`log_partition(ladder)`](@ref log_partition(::TrajectoryLadder)) and
[`log_likelihood(ladder, v)`](@ref log_likelihood(::TrajectoryLadder, ::AbstractArray)).
"""
mutable struct TrajectoryLadder{M, A <: AbstractArray, F <: AbstractVector}
    const rbm::M # model being trained (not a copy)
    const checkpoints::Vector{M} # frozen copies of `rbm` along its training trajectory
    const logZ::Vector{Float64} # log-partition functions of the checkpoints
    const chains::A # persistent chains of `rbm`
    chains_F::F # their free energies under the model they were last sampled from
    samples::A # equilibrium samples of the last checkpoint
    samples_F::F # their free energies under the last checkpoint
    reservoir::A # working copy of `samples`, exchanged with `chains`
    reservoir_F::F
    acceptance::Float64 # last swap acceptance between the last checkpoint and `rbm`
    since_checkpoint::Int # updates since the last checkpoint was frozen or restored
    rejections::Int # consecutive rejected updates
    τint::Float64 # integrated and exponential autocorrelation times (in sweeps) of the
    τexp::Float64 # chains' free energies, measured when building the last reservoir
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
    independent = _copy_model(rbm)
    independent.w .= 0
    samples = _default_fantasy_chains(independent, nreservoir) # exact samples
    samples_F = free_energy(independent, samples)
    chains = _default_fantasy_chains(independent, nchains)
    ladder = TrajectoryLadder(
        rbm, [independent], [Float64(log_partition_zero_weight(independent))],
        chains, free_energy(independent, chains), samples, samples_F, copy(samples),
        copy(samples_F), 1.0, 0, 0, NaN, NaN, sweeps, α, αmin
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
    F = free_energy(rbm, chains)
    return last(ladder.logZ) + _log_partition_ratio(
        last(ladder.checkpoints), samples, samples_F, rbm, chains, F; logw = chains_F - F
    )
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

Train an `RBM` with Parallel Trajectory Tempering (PTT; Béreux, Decelle, Furtlehner,
Seoane, arXiv:2607.27077).

Like [`pcd!`](@ref), but the persistent chains are kept at equilibrium by replica exchange
with frozen checkpoints of the training trajectory, held by a [`TrajectoryLadder`](@ref).
Each update runs `ladder.sweeps` sweeps, exchanging the chains with equilibrium samples of
the last checkpoint and then running `steps` Gibbs steps, before the gradient step. An
update that loses overlap with the last checkpoint is rejected: `rbm` is restored to that
checkpoint and the learning rate of the optimiser is halved.

`data` must have shape `(size(rbm.visible)..., nsamples)`.

# Keyword arguments
- `ladder`: the [`TrajectoryLadder`](@ref) of `rbm`, by default with
  `nchains = min(batchsize, nsamples)` chains.
- `steps::Int=1`: Gibbs steps per sweep.
- `optim::AbstractRule=Adam()`: optimizer rule from `Optimisers.jl`, whose learning rate
  can be halved. [`CossimDescent`](@ref) is the optimiser used in the paper.
- `callback=Returns(nothing)`: called after every update as
  `callback(; rbm, optim, state, ps, iter, vd, wd, ∂, vm, ladder)`, where `vm` are the
  chains of the ladder. Slurp unused keywords with a trailing `_...`.
- `batchsize`, `iters`, `wts`, `moments`, `l2_fields`, `l1_weights`, `l2_weights`,
  `l2l1_weights`, `zerosum`, `rescale`, `shuffle`, `ps`, `state`: as for [`pcd!`](@ref).

Returns `(state, ps)`.
"""
function ptt!(
        rbm::RBM,
        data::AbstractArray;
        batchsize::Int = 1,
        iters::Int = 1, # number of gradient updates
        wts::AbstractVector{<:Real} = uniform_wts(rbm.visible, data), # data weights
        steps::Int = 1, # Gibbs steps per sweep
        ladder::TrajectoryLadder = TrajectoryLadder(rbm; nchains = min(batchsize, size(data)[end]), steps),
        optim::AbstractRule = Adam(), # optimizer rule
        moments = moments_from_samples(rbm.visible, data; wts), # sufficient statistics for visible layer

        # regularization
        l2_fields::Real = 0, # visible fields L2 regularization
        l1_weights::Real = 0, # weights L1 regularization
        l2_weights::Real = 0, # weights L2 regularization
        l2l1_weights::Real = 0, # weights L2/L1 regularization

        # gauge
        zerosum::Bool = true, # zerosum gauge for Potts layers
        rescale::Bool = true, # normalize weights to unit norm (for continuous hidden units only)

        callback = Returns(nothing), # called for every batch

        shuffle::Bool = true,

        # parameters to optimize
        ps = (; visible = rbm.visible.par, hidden = rbm.hidden.par, w = rbm.w),
        state = setup(optim, ps),
    )
    ladder.rbm === rbm || throw(ArgumentError("the ladder was built for another model"))
    wts_mean, batchsize = _pcd_check_args(rbm, data, wts, batchsize)

    # initial gauge; zerosum! first because rescaling preserves the zero-sum gauge,
    # while zerosum! perturbs weight norms
    zerosum && zerosum!(rbm)
    rescale && rescale_weights!(rbm)

    for (iter, (vd, wd)) in zip(1:iters, infinite_minibatches(data, wts; batchsize, shuffle))
        # negative phase first, since a rejected update restores the parameters
        if _ptt_update!(ladder, rbm; steps) === :rejected
            ladder.rejections ≤ 30 || error("PTT lost equilibrium after 30 consecutive learning rate halvings")
            _halve_learning_rate!(state)
        end
        ∂m = ∂free_energy(rbm, ladder.chains)

        # positive phase
        ∂d = ∂free_energy(rbm, vd; wts = wd, moments)

        # weighted minibatch bias correction, in the gradient eltype
        batch_weight = convert(float(real(eltype(∂d.w))), mean(wd) / wts_mean)
        ∂ = (∂d - ∂m) * batch_weight

        # weight decay
        ∂regularize!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights, zerosum)

        # feed gradient to Optimiser rule
        gs = (; visible = ∂.visible, hidden = ∂.hidden, w = ∂.w)
        state, ps = update!(state, ps, gs)
        _validate_layer_parameters(rbm)

        # reset gauge (zerosum! first, as above)
        zerosum && zerosum!(rbm)
        rescale && rescale_weights!(rbm)

        callback(; rbm, optim, state, ps, iter, vd, wd, ∂, vm = ladder.chains, ladder)
    end
    return state, ps
end

#= One PTT update of the chains of `model`, which moved along its trajectory since the last
update. Returns `:rejected` if `model` lost overlap with the last checkpoint, in which case
`model` is restored to that checkpoint and the chains to samples from its reservoir;
`:frozen` if `model` was frozen as a new checkpoint; and `:accepted` otherwise.

The chains still sample the model before its last move, so the acceptance that decides
between these outcomes reweights them to `model`; otherwise a step too large would go
unnoticed until the chains catch up. The effective sample fraction of this reweighting
measures the overlap between the model before and after its last move: a value below
`αmin` (corresponding to a swap acceptance of about 0.3 for Gaussian log-weights) rejects
the move. =#
function _ptt_update!(ladder::TrajectoryLadder, model; steps::Int)
    status = :accepted
    for sweep in 1:ladder.sweeps
        proposal = _propose_exchange(ladder, model)
        if sweep == 1
            logw = ladder.chains_F - proposal.Fθx # reweights the chains to `model`
            w = exp.(logw .- maximum(logw))
            ladder.acceptance = sum(w .* min.(1, exp.(proposal.Δ))) / sum(w)
            ess = sum(w)^2 / sum(abs2, w) / length(w)
            status = ess < ladder.αmin ? :rejected : _ptt_status(ladder)
            if status === :rejected
                _copyto_model!(model, last(ladder.checkpoints))
                ladder.chains .= proposal.y
                ladder.chains_F .= proposal.Fy
                ladder.since_checkpoint = 0
                ladder.rejections += 1
                return status
            end
        end
        _exchange!(ladder, proposal)
        sweep == 1 && status === :frozen && _push_checkpoint!(ladder, model; steps)
        ladder.chains .= sample_v_from_v(model, ladder.chains; steps)
    end
    ladder.chains_F .= free_energy(model, ladder.chains)
    ladder.since_checkpoint += 1
    ladder.rejections = 0
    return status
end

function _ptt_status(ladder::TrajectoryLadder)
    ladder.acceptance ≥ ladder.α && return :accepted
    ladder.acceptance ≥ ladder.αmin && ladder.since_checkpoint > 1 && return :frozen
    return :rejected
end

#= Proposes to exchange each chain of `model` with a random reservoir member, an equilibrium
sample of the last checkpoint. `Δ` holds the log Metropolis ratios. =#
function _propose_exchange(ladder::TrajectoryLadder, model)
    x = ladder.chains
    idx = randperm(_nsamples(ladder.reservoir))[1:_nsamples(x)]
    y = ladder.reservoir[.., idx]
    Fy = ladder.reservoir_F[idx]
    Fx = free_energy(last(ladder.checkpoints), x)
    Fθx = free_energy(model, x)
    Δ = (Fθx - Fx) - (free_energy(model, y) - Fy)
    return (; idx, y, Fx, Fy, Fθx, Δ)
end

# applies the exchange `proposal` by the Metropolis rule; returns the accepted swaps
function _exchange!(ladder::TrajectoryLadder, proposal::NamedTuple)
    (; idx, y, Fx, Fy, Δ) = proposal
    accept = log.(rand!(similar(Δ))) .< Δ
    _swap!(ladder.chains, y, accept)
    ladder.reservoir[.., idx] = y
    ladder.reservoir_F[idx] = ifelse.(accept, Fx, Fy)
    return accept
end

# swaps the samples `x[.., n]` and `y[.., n]` where `accept[n]` holds
function _swap!(x::AbstractArray, y::AbstractArray, accept::AbstractVector)
    mask = reshape(accept, ntuple(Returns(1), ndims(x) - 1)..., length(accept))
    x′ = ifelse.(mask, y, x)
    y .= ifelse.(mask, x, y)
    x .= x′
    return nothing
end

#= Freezes a copy of `model` as a new checkpoint. Its chains are thermalized by exchanges
with the reservoir of the previous checkpoint, for 20 times the autocorrelation time of their
free energies (and at least `minsweeps` sweeps), and then collected every 2 integrated
autocorrelation times into new equilibrium samples. Exchanges with chains lagging behind a
moving model slightly bias the reservoir, which is therefore first reset to the equilibrium
samples of the previous checkpoint. =#
function _push_checkpoint!(ladder::TrajectoryLadder, model; steps::Int, minsweeps::Int = 20, maxsweeps::Int = 10_000)
    checkpoint = _copy_model(model)
    ladder.reservoir .= ladder.samples
    ladder.reservoir_F .= ladder.samples_F

    # returns the free energies of the chains under the checkpoint, before the sweep
    function sweep!()
        proposal = _propose_exchange(ladder, checkpoint)
        _exchange!(ladder, proposal)
        ladder.chains .= sample_v_from_v(checkpoint, ladder.chains; steps)
        return Array{Float64}(proposal.Fθx)
    end

    F = reduce(hcat, sweep!() for _ in 1:minsweeps)
    τint, τexp = _autocorrelation_times(F)
    while size(F, 2) < 20max(τint, τexp)
        size(F, 2) < maxsweeps || error("PTT failed to thermalize a new checkpoint within $maxsweeps sweeps")
        F = hcat(F, reduce(hcat, sweep!() for _ in axes(F, 2))) # doubles the run
        τint, τexp = _autocorrelation_times(F)
    end

    samples = similar(ladder.samples)
    nchains, nsamples = _nsamples(ladder.chains), _nsamples(samples)
    for n in 1:nchains:nsamples
        foreach(_ -> sweep!(), 1:ceil(Int, 2τint))
        m = min(nchains, nsamples - n + 1)
        samples[.., n:(n + m - 1)] .= view(ladder.chains, .., 1:m)
    end

    samples_F = free_energy(checkpoint, samples)
    logZ = last(ladder.logZ) + _log_partition_ratio(
        last(ladder.checkpoints), ladder.samples, ladder.samples_F, checkpoint, samples, samples_F
    )
    push!(ladder.checkpoints, checkpoint)
    push!(ladder.logZ, logZ)
    ladder.samples, ladder.samples_F = samples, samples_F
    ladder.reservoir, ladder.reservoir_F = copy(samples), copy(samples_F)
    ladder.chains_F .= free_energy(checkpoint, ladder.chains)
    ladder.τint, ladder.τexp = τint, τexp
    ladder.since_checkpoint = 0
    return ladder
end

#= Builds the initial ladder, from the independent-site model to `ladder.rbm`, along models
whose weights are those of `ladder.rbm` scaled by β ∈ [0, 1], in `nsteps` steps (halved when
rejected). `ladder.rbm` is frozen as the last checkpoint, so that rejected training updates
never move back further than the initial model. =#
function _anneal!(ladder::TrajectoryLadder; steps::Int, nsteps::Int)
    model = _copy_model(ladder.rbm)
    β = β₀ = 0.0 # current β, and β of the last checkpoint
    δ = 1 / nsteps
    while β < 1
        β = min(β + δ, 1.0)
        model.w .= β .* ladder.rbm.w
        status = _ptt_update!(ladder, model; steps)
        if status === :rejected
            β = β₀
            δ = δ / 2
            δ > 1.0e-6 || error("PTT failed to anneal from the independent-site model")
        elseif status === :frozen
            β₀ = β
        end
    end
    return β₀ == 1 ? ladder : _push_checkpoint!(ladder, ladder.rbm; steps)
end

#= Integrated and exponential autocorrelation times (in sweeps) of the time series `F`
(chains × sweeps), the free energies of the chains, which relax both through exchanges and
Gibbs sampling. The integrated time uses Sokal's self-consistent window. =#
function _autocorrelation_times(F::AbstractMatrix{<:Real})
    x = F .- mean(F)
    T = size(x, 2)
    C₀ = dot(x, x) / length(x)
    τint = 0.5
    τexp = 0.0
    for t in 1:(T ÷ 2)
        C = dot(view(x, :, (1 + t):T), view(x, :, 1:(T - t))) / (size(x, 1) * (T - t) * C₀)
        C > 0.05 && (τexp = max(τexp, -t / log(C)))
        τint += C
        t ≥ 6τint && break
    end
    return max(τint, 0.5), τexp
end

#= Bennett acceptance ratio estimate of log(Z₁ / Z₀) from equilibrium samples `x₀` of `m₀`,
with free energies `F₀x₀` under `m₀`, and samples `x₁` of `m₁`, with free energies `F₁x₁`
under `m₁` (Bennett, J. Comput. Phys. 22, 245 (1976); Shirts et al., Phys. Rev. Lett. 91,
140601 (2003)). Samples `x₁` can carry importance log-weights `logw`, entering through their
normalized weights and effective number. Solves the self-consistent equation for
Δf = log(Z₀ / Z₁) by bisection. =#
function _log_partition_ratio(m₀, x₀, F₀x₀, m₁, x₁, F₁x₁; logw = Zeros(length(F₁x₁)))
    W₀ = Array{Float64}(free_energy(m₁, x₀) - F₀x₀) # forward "work"
    W₁ = Array{Float64}(free_energy(m₀, x₁) - F₁x₁) # reverse "work"
    p₁ = softmax(Array{Float64}(logw))
    n₁ = 1 / sum(abs2, p₁) # effective number of samples x₁
    M = log(length(W₀) / n₁)
    g(Δf) = sum(w -> logistic(Δf - M - w), W₀) - n₁ * sum(p₁ .* logistic.(M .- W₁ .- Δf))
    lo = hi = -_logmeanexp(-W₀) # one-sided (exponential averaging) estimate
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
_logmeanexp(x::AbstractArray) = logsumexp(x) - log(length(x))
