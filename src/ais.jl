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

Addendum: I think Burda's paper has a typo. The correct expression for the weights In
reverse AIS (which I use here) can be found in Upadhya et al 2015, Equation 10
(https://link.springer.com/chapter/10.1007/978-3-319-26535-3_62).

Bonus: A discussion of estimating partition function in RBMs, comparing several algorithms:

https://www.sciencedirect.com/science/article/pii/S0004370219301948

For a variant or RAISE: https://arxiv.org/abs/1511.02543
=#

function ais_step(rbm0::RBM, rbm1::RBM, v::AbstractArray; β::Real)
    @assert 0 < β < 1
    rbm = anneal(rbm0, rbm1; β)
    F0 = free_energy(rbm, v)
    v1 = sample_v_from_v(rbm, v)
    F1 = free_energy(rbm, v1)
    return v1, F1 - F0
end

"""
    ais(rbm0, rbm1, v0, βs)

Provided `v0` is an unbiased sample from `rbm0`, returns `F` such that `mean(exp.(F))` is
an unbiased estimator of `Z1/Z0`, the ratio of partition functions of `rbm1` and `rbm0`.
"""
function ais(rbm0::RBM, rbm1::RBM, v::AbstractArray, βs::AbstractVector)
    @assert issorted(βs) && 0 ≤ first(βs) ≤ last(βs) ≤ 1
    F = free_energy(rbm0, v)
    for β in βs
        if iszero(β) || isone(β)
            continue
        else
            v, ΔF = ais_step(rbm0, rbm1, v; β)
            F += ΔF
        end
    end
    F -= free_energy(rbm1, v)
    return F
end

function ais(init::AbstractLayer, rbm1::RBM, v0::AbstractArray, βs::AbstractVector)
    rbm0 = anneal_zero(init, rbm1)
    return ais(rbm0, rbm1, v0, βs)
end

function ais(rbm0::Union{RBM,AbstractLayer}, rbm1::RBM, v0::AbstractArray; nbetas::Int=2)
    βs = range(0, 1, nbetas)
    return ais(rbm0, rbm1, v0, βs)
end

"""
    aise(rbm, [βs]; [nbetas], init=rbm.visible, nsamples=1)

AIS estimator of the log-partition function of `rbm`. It is recommended to fit `init` to
the single-site statistics of `rbm` (or the data).

!!! tip Use large `nbetas`
    For more accurate estimates, use larger `nbetas`. It is usually better to have
    large `nbetas` and small `nsamples`, rather than large `nsamples` and small `nbetas`.
"""
function aise(rbm::RBM, βs::AbstractVector{<:Real}; init::AbstractLayer=rbm.visible, nsamples::Int=1)
    rbm0 = anneal_zero(init, rbm)
    v0 = sample_from_inputs(init, Falses(size(init)..., nsamples))
    F = ais(rbm0, rbm, v0, βs)
    return F .+ log_partition_zero_weight(rbm0)
end

"""
    raise(rbm::RBM, βs; v, init)

Reverse AIS estimator of the log-partition function of `rbm`.
While `aise` tends to understimate the log of the partition function, `raise` tends to
overstimate it.
`v` must be an equilibrated sample from `rbm`.
"""
function raise(rbm::RBM, βs::AbstractVector; v::AbstractArray, init::AbstractLayer=rbm.visible)
    rbm0 = anneal_zero(init, rbm)
    F = ais(rbm, rbm0, v, βs)
    return log_partition_zero_weight(rbm0) .- F
end

aise(rbm::RBM; nbetas::Int=10000, kw...) = aise(rbm, range(0, 1, nbetas); kw...)
raise(rbm::RBM; nbetas::Int=10000, kw...) = raise(rbm, range(0, 1, nbetas); kw...)

"""
    anneal(rbm0, rbm1; β)

Returns an RBM that interpolates between `rbm0` and `rbm1`.
Denoting by `E0(v, h)` and `E1(v, h)` the energies assigned by `rbm0` and `rbm1`,
respectively, the returned RBM assigns energies given by:

    E(v,h) = (1 - β) * E0(v) + β * E1(v, h)
"""
function anneal(rbm0::RBM, rbm1::RBM; β::Real)
    vis = anneal(rbm0.visible, rbm1.visible; β)
    hid = anneal(rbm0.hidden, rbm1.hidden; β)
    w = (1 - β) * rbm0.w + β * rbm1.w
    return RBM(vis, hid, w)
end

function anneal(init::AbstractLayer, rbm1::RBM; β::Real)
    rbm0 = anneal_zero(init, rbm1)
    return anneal(rbm0, rbm1; β)
end

anneal_zero(init::AbstractLayer, rbm1::RBM) = RBM(init, anneal(rbm1.hidden; β=0), Zeros(rbm1.w))

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
log_partition_zero_weight(rbm::RBM) = -free_energy(rbm.visible) - free_energy(hidden(rbm))

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
