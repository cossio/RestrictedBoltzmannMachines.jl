#=
Rejection sampler based on algorithm from Robert (1995)

  - Available at http://arxiv.org/abs/0907.4010

Implementation from Distributions.jl, with few modifications to make it more generic.
Also here we avoid the overhead of computing the CDF, which `truncated(Normal(), a, b)`
always does when it constructs the truncated distribution (`tp`).
We specialize to the case where `b = Inf`.
=#

"""
    randnt([rng], a)

Random standard normal lower truncated at `a` (that is, Z ≥ a).
"""
randnt(rng::AbstractRNG, a::Real) = randnt(rng, float(a))
randnt(rng::AbstractRNG, a::BigFloat) = randnt(rng, Float64(a))
randnt(a::Real) = randnt(default_rng(), a)

# This union mirrors private `Base.IEEEFloat` without relying on a non-public name.
function randnt(rng::AbstractRNG, a::Union{Float16, Float32, Float64})
    local r = a
    if a ≤ 0
        while true
            r = randn(rng, typeof(a))
            r ≥ a && break
        end
    elseif a < Inf
        t = sqrt1half(a)
        while true
            r = a + randexp(rng, typeof(a)) / t
            u = rand(rng, typeof(a))
            u < exp(-(r - t)^2 / 2) && break
        end
    end
    return r
end

"""
    sqrt1half(x)

Accurate computation of sqrt(1 + (x/2)^2) + |x|/2.
"""
function sqrt1half(x::Real)
    h = x / 2
    return hypot(one(h), h) + abs(h)
end

"""
    randnt_half([rng], μ, σ)

Samples the normal distribution with mean `μ` and standard deviation `σ`
truncated to positive values.
"""
function randnt_half(rng::AbstractRNG, μ::Real, σ::Real)
    z = randnt(rng, -μ / σ)
    return μ + σ * z
end

randnt_half(μ::Real, σ::Real) = randnt_half(default_rng(), μ, σ)

"""
    tnmean(a)

Mean of the standard normal distribution,
truncated to the interval (a, +∞).
"""
function tnmean(a::Real)
    two = oftype(a, 2)
    return sqrt(two / π) / erfcx(a / sqrt(two))
end

"""
    tnvar(a)

Variance of the standard normal distribution,
truncated to the interval (a, +∞).
"""
tnvar(a::Real) = last(tnmeanvar(a))

"""
    tnmeanvar(a)

Mean and variance of the standard normal distribution truncated to the interval (a, +∞).
Equivalent to `tnmean(a), tnvar(a)` but saves some common computations.
"""
function tnmeanvar(a::Real)
    levels = _tn_tail_levels(a)
    if levels > 0
        δ, ν = _tnmeanvar_tail(a, levels)
        return a + δ, ν
    end
    μ = tnmean(a)
    ν = one(μ) - (μ - a) * μ
    return μ, ν
end

"""
    tnmeanvar(μ, σ)

Mean and variance of a normal distribution with mean `μ` and standard deviation
`σ > 0`, truncated to positive values. Computes the shifted tail mean directly
to avoid cancellation when `μ` is negative relative to `σ`.
"""
function tnmeanvar(μ::Real, σ::Real)
    a = -μ / σ
    levels = _tn_tail_levels(a)
    if levels > 0
        δ, ν = _tnmeanvar_tail(a, levels)
        return σ * δ, σ^2 * ν
    end
    tμ = tnmean(a)
    tν = one(tμ) - (tμ - a) * tμ
    return μ + σ * tμ, σ^2 * tν
end

# broadcasts and returns a tuple of arrays, rather than an array of tuples like tnmeanvar.(A)
tnmeanvar(a::AbstractArray) = _unzip(tnmeanvar.(a))

#=
The excess mean δ = E[Z | Z > a] - a is the first of the ratios T_k = m_k / m_{k-1} of
consecutive moments m_k = E[(Z - a)^k | Z > a], which satisfy the backward recurrence

    T_k = k / (a + T_{k+1}),

that is, Laplace's continued fraction δ = 1 / (a + 2 / (a + 3 / (a + ...))). The variance
is the first of V_k = -dT_k/da, which follow by differentiating the recurrence:

    V_k = T_k^2 (1 - V_{k+1}) / k.

For a > 0 the recurrence only adds, multiplies and divides positive numbers (0 < V_k < 1),
whereas `1 - (μ - a) * μ` with `μ = tnmean(a)` cancels: it loses a factor of about `a^2` in
the excess mean and `a^4` in the variance.

`_tn_tail_levels(a)` is the number of levels of the recurrence that `_tnmeanvar_tail` needs
at the precision of `a`, or zero where the direct formula is used instead. The recurrence
damps the error of its starting value by about `exp(-2a√levels)`, so the levels needed grow
as `a` decreases, and with the precision. The count was fitted to truncation errors below
`eps / 8` in Float32 and Float64, and exceeds what higher precisions need. The recurrence is
used where it takes at most 22 levels, which cost up to about 1.5 times the direct formula:
from `a = 1` in Float32 and `a ≈ 3.9` in Float64. It is always used for `a ≥ 8`, and never
below `a = 1`, where the direct formula loses little. The count is a float, because
converting an integer inside `_tnmeanvar_tail` makes the kernel about twice slower in a
broadcast (the successive elements are chained through the register that receives the
conversion).
=#
function _tn_tail_levels(a::Real)
    x = float(a)
    levels = 2 + ceil(oftype(x, precision(x)^2) / 36 / min(x, 16))
    return ifelse(x ≥ 1 && (x ≥ 8 || levels ≤ 22), levels, zero(levels))
end

function _tnmeanvar_tail(a::Real, levels::Real)
    #=
    The recurrence starts at k = levels + 1 from the first terms of the expansion of T_k and
    V_k for large s^2 = a^2 + 4k, in powers of e = 1 / s^2 with coefficients polynomial in
    τ = a / s:

        T_k ≈ (s - a) / 2 * g,    g = 1 - e + e^2 / 2 * (5τ + 1 - e * (30τ^2 + 5τ - 5)),
        V_k ≈ (1 - τ) / 2 * h,    h = 1 - (2τ + 1) * e
                                      + e^2 / 2 * (25τ^2 + 9τ - 4 - e * (240τ^3 + 65τ^2 - 85τ - 10)).

    The leading term solves T (a + T) = k. g follows by substituting in T_k (a + T_{k+1}) = k
    and matching the powers of e, and h from g by differentiation with respect to a. Beyond
    a = 1 / eps the levels above the first do not change the result, and the bound keeps the
    square finite.
    =#
    b = min(float(a), inv(eps(one(float(a)))))
    k = oftype(b, levels) + 1
    s2 = b^2 + 4k
    s = sqrt(s2)
    e = inv(s2)
    τ = b * (s * e)
    g = muladd(e^2 / 2, muladd(-e, evalpoly(τ, (-5, 5, 30)), 5τ + 1), 1 - e)
    h = muladd(
        e^2 / 2, muladd(-e, evalpoly(τ, (-10, -85, 65, 240)), evalpoly(τ, (-4, 9, 25))),
        muladd(-e, 2τ + 1, 1)
    )
    #=
    Levels k = levels, ..., 2 without divisions, with T_k = s * N / D and V_k = W / D^2. The
    scaling by s keeps D in (0, 2). Since (1 - τ) / 2 = 2k * e / (1 + τ), the starting values
    are:
    =#
    D = 1 + τ
    N = 2k * e * g
    W = 2k * e * h * D
    while k > 2
        k -= 1
        ke = k * e
        M = ke * D
        W = muladd(-ke, W, M * D)
        D = muladd(τ, D, N)
        N = M
    end
    # The first level in the original variables, so that the scaling does not add to its error.
    r = inv(D)
    δ = inv(a + s * N * r)
    ν = δ^2 * (1 - W * r * r)
    return δ, ν
end
