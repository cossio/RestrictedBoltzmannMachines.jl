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
    v = transfer_sample(init, Falses(size(init)..., nsamples)) # v[0]
    F_curr = free_energy(anneal(init, rbm; β = 0), v) # F[0](v[0])
    annealed_rbm = anneal(init, rbm; β = 1 / nbetas) # β[0] -> β[1]
    F_next = free_energy(annealed_rbm, v) # F[k](v[k - 1])
    R = F_curr - F_next
    for β in (2:nbetas) / nbetas
        v = sample_v_from_v(annealed_rbm, v) # v[k-1] -> v[k]
        F_curr = free_energy(annealed_rbm, v) # F[k](v[k])
        annealed_rbm = anneal(init, rbm; β) # β[k] -> β[k + 1]
        F_next = free_energy(annealed_rbm, v) # F[k + 1](v[k])
        R += F_curr - F_next
    end
    R .+= log_partition_zero_weight(anneal(init, rbm; β = 0))
    return R # return vector of samples; use logmeanexp(R) to get a single estimate
end

function rais(rbm::RBM, v::AbstractArray; nbetas::Int, init::AbstractLayer = visible(rbm))
    nsamples = size(v)[end]
    @assert size(v) == (size(visible(rbm))..., nsamples)
    @assert size(init) == size(visible(rbm))

    F_curr = free_energy(rbm, v) # F(v[0])
    annealed_rbm = anneal(init, rbm; β = (nbetas - 1) / nbetas) # β[K] -> β[K - 1]
    F_next = free_energy(annealed_rbm, v) # F(v[1])
    R = F_curr - F_next

    for β in reverse((1:(nbetas - 2)) / nbetas)
        v = sample_v_from_v(annealed_rbm, v) # v[k + 1] -> v[k]
        F_curr = free_energy(annealed_rbm, v) # F[k+1](v[k])
        annealed_rbm = anneal(init, rbm; β) # β[k + 1] -> β[k]
        F_next = free_energy(annealed_rbm, v) # F[k](v[k])
        R += F_curr - F_next
    end
    R .-= log_partition_zero_weight(anneal(init, rbm; β = 0))
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
function logmeanexp(A::AbstractArray; dims=:)
    R = logsumexp(A; dims)
    N = length(A) ÷ length(R)
    return R .- log(N)
end

"""
    logvarexp(A; dims=:)

Computes `log.(var(exp.(A); dims))`, in a numerically stable way.
"""
function logvarexp(
    A::AbstractArray; dims=:, corrected::Bool=true, logmean=logmeanexp(A; dims)
)
    R = logsumexp(2logsubexp.(A, logmean); dims)
    N = length(A) ÷ length(R)
	if corrected
		return R .- log(N - 1)
    else
        return R .- log(N)
    end
end

"""
    logstdexp(A; dims=:)

Computes `log.(std(exp.(A); dims))`, in a numerically stable way.
"""
function logstdexp(
    A::AbstractArray; dims=:, corrected::Bool=true, logmean=logmeanexp(A; dims)
)
    return logvarexp(A; dims, corrected, logmean) / 2
end
