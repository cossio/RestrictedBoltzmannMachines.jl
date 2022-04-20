#=
Annealed importance sampling (AIS) to estimate the partition function (and hence
the log-likelihood).
A nice explanation of AIS in general can be found in Goodfellow et al Deep Learning book.
Salakhutdinov et al (10.1145/1390156.1390266, http://www.cs.utoronto.ca/~rsalakhu/papers/bm.pdf)
discuss AIS for RBMs specifically.

AIS tends to ovestimate likelihoods.
In contrast, the Reverse AIS estimator (RAISE) tends to give underestimates.
Combining the two we obtain stochastic "sandwiches" containing the true value.
RAISE is introduced here: http://proceedings.mlr.press/v38/burda15.html
Using both, we have an interval where true log-likelihood lies.

Bonus: A discussion of estimating partition function in RBMs comparing several algorithms:

https://www.sciencedirect.com/science/article/pii/S0004370219301948
=#

"""
    ais(rbm; nbetas = 10000, nsamples = 100, init = visible(rbm))

Estimates the log-partition function of `rbm` by Annealed Importance Sampling (AIS).
Here `init` is an initial independent-site distribution, represented by a visible layer.
"""
function ais(rbm::RBM; nbetas::Int = 10000, nsamples::Int = 10, init::AbstractLayer = visible(rbm))
    @assert size(init) == size(visible(rbm))
    v = transfer_sample(init, Falses(size(init)..., nsamples))
    annealed_rbm = anneal(init, rbm; β = 0)
    lR = fill(log_partition_zero_weight(annealed_rbm), nsamples)
    for β in ((1:nbetas) .// nbetas)
        v = sample_v_from_v(annealed_rbm, v)
        F_prev = free_energy(annealed_rbm, v)
        annealed_rbm = anneal(init, rbm; β)
        F_curr = free_energy(annealed_rbm, v)
        lR += F_prev - F_curr
    end
    return logsumexp(lR) - log(nsamples)
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
    return RBM(vis, hid, β * weights(final))
    #return RBM(vis, hidden(final), β * weights(final))
end

anneal(layer::Binary; β::Real) = Binary(β * layer.θ)
anneal(layer::Spin; β::Real) = Spin(β * layer.θ)
anneal(layer::Potts; β::Real) = Potts(β * layer.θ)
anneal(layer::Gaussian; β::Real) = Gaussian(β * layer.θ, layer.γ)
anneal(layer::ReLU; β::Real) = ReLU(β * layer.θ, layer.γ)
anneal(layer::dReLU; β::Real) = dReLU(β * layer.θp, β * layer.θn, layer.γp, layer.γn)
anneal(layer::pReLU; β::Real) = pReLU(β * layer.θ, layer.γ, β * layer.Δ, layer.η)
anneal(layer::xReLU; β::Real) = xReLU(β * layer.θ, layer.γ, β * layer.Δ, layer.ξ)

anneal(init::Binary, final::Binary; β::Real) = Binary((1 - β) * init.θ + β * final.θ)
anneal(init::Spin, final::Spin; β::Real) = Spin((1 - β) * init.θ + β * final.θ)
anneal(init::Potts, final::Potts; β::Real) = Potts((1 - β) * init.θ + β * final.θ)

anneal(init::Gaussian, final::Gaussian; β::Real) = Gaussian(
    (1 - β) * init.θ + β * final.θ, final.γ
)

anneal(init::ReLU, final::ReLU; β::Real) = ReLU(
    (1 - β) * init.θ + β * final.θ, final.γ
)

anneal(init::dReLU, final::dReLU; β::Real) = dReLU(
    (1 - β) * init.θp + β * final.θp,
    (1 - β) * init.θn + β * final.θn,
    final.γp, final.γn
)

anneal(init::pReLU, final::pReLU; β::Real) = pReLU(
    (1 - β) * init.θ + β * final.θ,
    final.γ,
    (1 - β) * init.Δ + β * final.Δ,
    final.η
)

anneal(init::xReLU, final::xReLU; β::Real) = xReLU(
    (1 - β) * init.θ + β * final.θ,
    final.γ,
    (1 - β) * init.Δ + β * final.Δ,
    final.ξ
)

"""
    log_partition_zero_weight(rbm)

Log-partition function of a zero-weight version of `rbm`.
"""
log_partition_zero_weight(rbm::RBM) = -free_energy(visible(rbm)) - free_energy(hidden(rbm))
