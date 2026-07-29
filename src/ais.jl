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

"""
    ais(rbm0, rbm1, v0, βs; steps=1)

Provided `v0` is an equilibrated sample from `rbm0`, returns `F` such that `mean(exp.(F))` is
an unbiased estimator of `Z1/Z0`, the ratio of partition functions of `rbm1` and `rbm0`.
`steps` is the number of Gibbs sweeps performed at each intermediate temperature.

!!! tip Use [`logmeanexp`](@ref)
    `logmeanexp(F)`, using the function `logmeanexp`[@ref] provided in this package,
    tends to give a better approximation of `log(Z1) - log(Z0)` than `mean(F)`.
"""
function ais(rbm0::RBM, rbm1::RBM, v::AbstractArray, βs::AbstractVector; steps::Int = 1)
    @assert issorted(βs) && 0 == first(βs) ≤ last(βs) == 1
    F = free_energy(rbm0, v)
    for β in βs
        if iszero(β) || isone(β)
            continue
        else
            rbm = anneal(rbm0, rbm1; β)
            F -= free_energy(rbm, v)
            v = sample_v_from_v(rbm, v; steps)
            F += free_energy(rbm, v)
        end
    end
    F -= free_energy(rbm1, v)
    return F
end

function ais(rbm0::RBM, rbm1::RBM, v0::AbstractArray; nbetas::Int = 2, steps::Int = 1)
    βs = range(0, 1, nbetas)
    return ais(rbm0, rbm1, v0, βs; steps)
end

"""
    aise(rbm, [βs]; [nbetas], init=rbm.visible, nsamples=1, steps=1)

AIS estimator of the log-partition function of `rbm`. It is recommended to fit `init` to
the single-site statistics of `rbm` (or the data).

!!! tip Use large `nbetas`
    For more accurate estimates, use larger `nbetas`. It is usually better to have
    large `nbetas` and small `nsamples`, rather than large `nsamples` and small `nbetas`.

!!! tip Adaptive schedules
    Instead of the default uniform grid of `nbetas` temperatures, an adapted schedule
    computed by [`adaptive_betas`](@ref) can be passed as `βs`.
"""
function aise(rbm::RBM, βs::AbstractVector{<:Real}; init::AbstractLayer = rbm.visible, nsamples::Int = 1, steps::Int = 1)
    rbm0 = anneal_zero(init, rbm)
    v0 = sample_from_inputs(init, Falses(size(init)..., nsamples))
    F = ais(rbm0, rbm, v0, βs; steps)
    return F .+ log_partition_zero_weight(rbm0)
end

"""
    raise(rbm::RBM, [βs]; [nbetas], v, init=rbm.visible, steps=1)

Reverse AIS estimator of the log-partition function of `rbm`.
While `aise` tends to understimate the log of the partition function, `raise` tends to
overstimate it. `v` must be an equilibrated sample from `rbm`.

`βs` follows the same convention as [`aise`](@ref): inverse temperatures of the target
`rbm`, sorted from 0 to 1. The annealing path traverses it in reverse (from `rbm` down to
the reference), so the same schedule (e.g. from [`adaptive_betas`](@ref)) can be shared
between `aise` and `raise`.

!!! tip Use [`logmeanexp`](@ref)
    If `F = raise(...)`, then `-logmeanexp(-F)`, using the function `logmeanexp`[@ref]
    provided in this package, tends to give a better approximation of `log(Z)` than `mean(F)`.

!!! tip Sandwiching the log-partition function
    If `Rf = aise(...)`, `Rr = raise(...)` are the AIS and reverse AIS estimators, we have the
    stochastic bounds `logmeanexp(Rf) ≤ log(Z) ≤ -logmeanexp(-Rr)`.
"""
function raise(rbm::RBM, βs::AbstractVector; v::AbstractArray, init::AbstractLayer = rbm.visible, steps::Int = 1)
    rbm0 = anneal_zero(init, rbm)
    F = ais(rbm, rbm0, v, 1 .- reverse(βs); steps)
    return log_partition_zero_weight(rbm0) .- F
end

aise(rbm::RBM; nbetas::Int = 10000, kw...) = aise(rbm, range(0, 1, nbetas); kw...)
raise(rbm::RBM; nbetas::Int = 10000, kw...) = raise(rbm, range(0, 1, nbetas); kw...)

"""
    adaptive_betas(rbm; init=rbm.visible, nsamples=100, target=0.99, max_betas=10_000, min_increment=1e-6, steps=1)
    adaptive_betas(rbm0, rbm1, v0; target=0.99, max_betas=10_000, min_increment=1e-6, steps=1)

Adaptively selects an inverse temperature schedule `βs` for annealed importance sampling,
placing more intermediate temperatures where the annealing path from the reference model
to the target model is hard, and fewer where it is easy.

Starting from `β = 0`, a pilot population of Monte Carlo chains is annealed towards
`β = 1`. Each next inverse temperature is chosen (by bisection) as the largest `β` whose
incremental importance weights retain a normalized effective sample size of at least
`target` over the pilot population. `min_increment` bounds the bisection tolerance and
the smallest allowed temperature step, and `max_betas` caps the schedule length (if the
cap is reached, the last temperature is replaced by 1 and a warning is emitted).

In the first form, the reference model and the pilot population are constructed from
`init` as in [`aise`](@ref). In the second form, `v0` must be an equilibrated sample
from `rbm0`, as in [`ais`](@ref).

Returns a sorted vector of inverse temperatures, starting at 0 and ending at 1, which can
be passed as `βs` to [`ais`](@ref), [`aise`](@ref), or [`raise`](@ref). Since the
schedule is adapted on a fresh pilot population, the returned `βs` can be used in
subsequent AIS runs without biasing them.
"""
function adaptive_betas(
        rbm0::RBM, rbm1::RBM, v::AbstractArray;
        target::Real = 0.99, max_betas::Int = 10_000, min_increment::Real = 1.0e-6, steps::Int = 1
    )
    @assert 0 < target < 1
    @assert 0 < min_increment < 1
    @assert max_betas ≥ 2
    # need at least two chains to estimate the effective sample size
    @assert length(v) ≥ 2 * length(rbm0.visible)
    β = 0.0
    βs = [β]
    F = vec(free_energy(rbm0, v))
    while β < 1 && length(βs) < max_betas
        β = next_beta(rbm0, rbm1, v, F, β; target, min_increment)
        push!(βs, β)
        if β < 1
            rbm = anneal(rbm0, rbm1; β)
            v = sample_v_from_v(rbm, v; steps)
            F = vec(free_energy(rbm, v))
        end
    end
    if last(βs) < 1
        @warn "adaptive_betas reached max_betas = $max_betas before β = 1; consider lowering `target` or raising `max_betas`"
        βs[end] = 1.0
    end
    return βs
end

function adaptive_betas(rbm::RBM; init::AbstractLayer = rbm.visible, nsamples::Int = 100, kw...)
    rbm0 = anneal_zero(init, rbm)
    v0 = sample_from_inputs(init, Falses(size(init)..., nsamples))
    return adaptive_betas(rbm0, rbm, v0; kw...)
end

#=
Largest β′ ∈ (β, 1] such that the normalized effective sample size of the incremental
importance weights exp.(F - F_β′(v)) stays above `target`, found by bisection
(the ESS is decreasing in β′, up to Monte Carlo noise).
=#
function next_beta(
        rbm0::RBM, rbm1::RBM, v::AbstractArray, F::AbstractVector, β::Real;
        target::Real, min_increment::Real
    )
    ess(β′) = incremental_ess(F, vec(free_energy(anneal(rbm0, rbm1; β = β′), v)))
    if ess(1.0) ≥ target
        return 1.0
    end
    lo, hi = β, 1.0
    while hi - lo > min_increment
        mid = (lo + hi) / 2
        if ess(mid) ≥ target
            lo = mid
        else
            hi = mid
        end
    end
    return min(max(lo, β + min_increment), 1.0) # always make progress
end

#=
Normalized effective sample size (ESS / number of chains) of the incremental importance
weights exp.(F0 - F1), computed in a numerically stable way.
=#
function incremental_ess(F0::AbstractVector, F1::AbstractVector)
    ℓ = F0 - F1
    return exp(2logsumexp(ℓ) - logsumexp(2ℓ) - log(length(ℓ)))
end

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

# Since every named layer parameter is a row of `par`, all layers anneal the same way.
function anneal(init::AbstractLayer, final::AbstractLayer; β::Real)
    @assert nameof(typeof(init)) === nameof(typeof(final))
    return _construct_like(init, (1 - β) * init.par + β * final.par)
end

anneal_zero(init::AbstractLayer, rbm::RBM) = RBM(init, anneal_zero(rbm.hidden), Zeros(rbm.w))

anneal_zero(l::Binary) = Binary(; θ = zero(l.θ))
anneal_zero(l::Spin) = Spin(; θ = zero(l.θ))
anneal_zero(l::Potts) = Potts(; θ = zero(l.θ))
anneal_zero(l::PottsGumbel) = PottsGumbel(; θ = zero(l.θ))
anneal_zero(l::Gaussian) = Gaussian(; θ = zero(l.θ), l.γ)
anneal_zero(l::ReLU) = ReLU(; θ = zero(l.θ), l.γ)
anneal_zero(l::dReLU) = dReLU(; θp = zero(l.θp), θn = zero(l.θn), l.γp, l.γn)
anneal_zero(l::pReLU) = pReLU(; θ = zero(l.θ), l.γ, Δ = zero(l.Δ), l.η)
anneal_zero(l::xReLU) = xReLU(; θ = zero(l.θ), l.γ, Δ = zero(l.Δ), l.ξ)
anneal_zero(l::nsReLU) = nsReLU(; θ = zero(l.θ), Δ = zero(l.Δ), l.ξ)

"""
    log_partition_zero_weight(rbm)

Log-partition function of a zero-weight version of `rbm`.
"""
log_partition_zero_weight(rbm) = cgf(rbm.visible) + cgf(rbm.hidden)

"""
    logmeanexp(A; dims=:)

Computes `log.(mean(exp.(A); dims))`, in a numerically stable way.
"""
function logmeanexp(A::AbstractArray; dims = :)
    R = logsumexp(A; dims)
    N = length(A) ÷ length(R)
    return R .- log(N)
end

"""
    logvarexp(A; dims=:)

Computes `log.(var(exp.(A); dims))`, in a numerically stable way.
"""
function logvarexp(
        A::AbstractArray; dims = :, corrected::Bool = true, logmean = logmeanexp(A; dims)
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
        A::AbstractArray; dims = :, corrected::Bool = true, logmean = logmeanexp(A; dims)
    )
    return logvarexp(A; dims, corrected, logmean) / 2
end
