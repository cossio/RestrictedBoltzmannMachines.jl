#=
Annealed importance sampling (AIS) to estimate the partition function (and hence
the log-likelihood).
A nice explanation of AIS in general can be found in Goodfellow et al Deep Learning book.
Salakhutdinov et al (10.1145/1390156.1390266, http://www.cs.utoronto.ca/~rsalakhu/papers/bm.pdf)
discusses AIS for RBMs specifically.

AIS tends to understimate the log of the partition function (in probability).
In contrast, Reverse AIS estimator (RAISE) can be used to obtain a stochastic upper bound.
See http://proceedings.mlr.press/v38/burda15.html.
Combining the two we can "sandwiches" the true value to have an idea if the Monte Carlo
chains have converged.

Bonus: A discussion of estimating partition function in RBMs, comparing several algorithms:

https://www.sciencedirect.com/science/article/pii/S0004370219301948

For a variant or RAISE: https://arxiv.org/abs/1511.02543
=#

"""
    ais(rbm; nbetas = 10000, nsamples = 10, init = visible(rbm))

Estimates the log-partition function of `rbm` by Annealed Importance Sampling (AIS).
Here `init` is an initial independent-site distribution, represented by a visible layer.
Returns `nsamples` variates, which can be averaged.

If `R = ais(rbm)`, then `mean(exp.(R))` is an unbiased estimator of the partition function.
Therefore `logmeanexp(R)` (defined in this package) can be used to estimate the
log-partition function. `std(R)` is a measure of the error in the estimate.
"""
function ais(rbm::RBM; nbetas::Int = 1000, nsamples::Int = 1, init::AbstractLayer = visible(rbm))
    @assert size(init) == size(visible(rbm))
    v = transfer_sample(init, Falses(size(init)..., nsamples))
    annealed_rbm = anneal(init, rbm; β = 0)
    R = fill(log_partition_zero_weight(annealed_rbm), nsamples)
    for β in ((1:nbetas) ./ nbetas)
        v = sample_v_from_v(annealed_rbm, v)
        F_prev = free_energy(annealed_rbm, v)
        annealed_rbm = anneal(init, rbm; β)
        F_curr = free_energy(annealed_rbm, v)
        R += F_prev - F_curr
    end
    return R # logmeanexp(R) to get estimate
end

function log_partition_ais_err(rbm::RBM; width::Real = 1, kwargs...)
    R = ais(rbm; kwargs...)
    lZ = logmeanexp(R)
    lσ = logvarexp(R) / 2
    hi = logaddexp(lZ, lσ + log(width))
    lo = logsubexp(lZ, lσ + log(width))
    return (lZ, lo, hi)
end

function raise(rbm::RBM, v::AbstractArray; nbetas::Int, init::AbstractLayer = visible(rbm))
    nsamples = size(v)[end]
    @assert size(v) == (size(visible(rbm))..., nsamples)
    @assert size(init) == size(visible(rbm))
    # init with -log(Z(β = 0))
    R = fill(-log_partition_zero_weight(anneal(init, rbm; β = 0)), nsamples)
    prev_rbm = rbm # rbm at temperature β[k + 1]
    for β in reverse((0:(nbetas - 1)) ./ nbetas)
        curr_rbm = anneal(init, rbm; β) # rbm at temperature β[k]
        v = sample_v_from_v(curr_rbm, v)
        F_prev = free_energy(prev_rbm, v) # p(v; β[k + 1])
        F_curr = free_energy(curr_rbm, v) # p(v; β[k])
        prev_rbm = curr_rbm
        R += F_prev - F_curr # accumulate log(p(v; β[k]) / p(v; β[k + 1]))
    end
    return R # -logmeanexp(R) to get estimate of Z
end

"""
    anneal(visible_init, rbm_final; β)

Returns an RBM that interpolates between an independent-site model `visible_init`
and `rbm`. Denoting by `E0(v)` the energies assigned by `visible_init`, and by
`E1(v, h)` the energies assigned by `rbm_final`, the returned RBM assigns energies
given by:

    E(v,h) = (1 - β) * E0(v) + β * E1(v, h)
"""
function anneal(init::AbstractLayer, final::RBM; β::Real)
    vis = anneal(init, visible(final); β)
    hid = anneal(hidden(final); β)
    w = oftype(weights(final), β * weights(final))
    return RBM(vis, hid, w)
    #return RBM(vis, hidden(final), β * weights(final))
end

anneal(layer::Binary; β::Real) = Binary(
    oftype(layer.θ, β * layer.θ)
)
anneal(layer::Spin; β::Real) = Spin(
    oftype(layer.θ, β * layer.θ)
)
anneal(layer::Potts; β::Real) = Potts(
    oftype(layer.θ, β * layer.θ)
)
anneal(layer::Gaussian; β::Real) = Gaussian(
    oftype(layer.θ, β * layer.θ), layer.γ
)
anneal(l::ReLU; β::Real) = ReLU(
    oftype(l.θ, β * l.θ), l.γ
)
anneal(layer::dReLU; β::Real) = dReLU(
    oftype(layer.θp, β * layer.θp),
    oftype(layer.θn, β * layer.θn),
    layer.γp, layer.γn
)
anneal(layer::pReLU; β::Real) = pReLU(
    oftype(layer.θ, β * layer.θ), layer.γ,
    oftype(layer.Δ, β * layer.Δ), layer.η
)
anneal(layer::xReLU; β::Real) = xReLU(
    oftype(layer.θ, β * layer.θ), layer.γ,
    oftype(layer.Δ, β * layer.Δ), layer.ξ
)
anneal(init::Binary, final::Binary; β::Real) = Binary(
    oftype(final.θ, (1 - β) * init.θ + β * final.θ)
)
anneal(init::Spin, final::Spin; β::Real) = Spin(
    oftype(final.θ, (1 - β) * init.θ + β * final.θ)
)
anneal(init::Potts, final::Potts; β::Real) = Potts(
    oftype(final.θ, (1 - β) * init.θ + β * final.θ)
)
anneal(init::Gaussian, final::Gaussian; β::Real) = Gaussian(
    oftype(final.θ, (1 - β) * init.θ + β * final.θ), final.γ
)
anneal(init::ReLU, final::ReLU; β::Real) = ReLU(
    oftype(final.θ, (1 - β) * init.θ + β * final.θ), final.γ
)
anneal(init::dReLU, final::dReLU; β::Real) = dReLU(
    oftype(final.θp, (1 - β) * init.θp + β * final.θp),
    oftype(final.θn, (1 - β) * init.θn + β * final.θn),
    final.γp, final.γn
)
anneal(init::pReLU, final::pReLU; β::Real) = pReLU(
    oftype(final.θ, (1 - β) * init.θ + β * final.θ), final.γ,
    oftype(final.Δ, (1 - β) * init.Δ + β * final.Δ), final.η
)
anneal(init::xReLU, final::xReLU; β::Real) = xReLU(
    oftype(final.θ, (1 - β) * init.θ + β * final.θ), final.γ,
    oftype(final.Δ, (1 - β) * init.Δ + β * final.Δ), final.ξ
)

"""
    log_partition_zero_weight(rbm)

Log-partition function of a zero-weight version of `rbm`.
"""
log_partition_zero_weight(rbm::RBM) = -free_energy(visible(rbm)) - free_energy(hidden(rbm))

"""
    logmeanexp(A; dims=:)

Computes `log.(mean(exp.(A); dims))`, in a numerically stable way.
"""
logmeanexp(A; dims=:) = logsumexp(A; dims) .- log(prod(size(A)[dims]))

"""
    logvarexp(A; dims=:)

Computes `log.(var(exp.(A); dims))`, in a numerically stable way.
"""
function logvarexp(A; dims=:, corrected::Bool=true, logμ = logmeanexp(A; dims))
	if corrected
		N = prod(size(A)[dims]) - 1
    else
        N = prod(size(A)[dims])
    end
	return logsumexp(2logsubexp.(A, logμ); dims) .- log(N)
end
