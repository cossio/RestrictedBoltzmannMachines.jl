"""
    normcdf(x)

Probablity that Z ≤ x, where Z is a standard normal sample_from_inputs variable.
"""
normcdf(x::Real) = erfc(-x / √two(x)) / 2

"""
    normcdf(a, b)

Probablity that a ≤ Z ≤ b, where Z is a standard normal sample_from_inputs variable.
WARNING: Silently returns a negative value if a > b.
"""
normcdf(a::Real, b::Real) = erf(a / √two(a), b / √two(b)) / 2

"""
    normcdfinv(x)

Inverse of normcdf.
"""
normcdfinv(x::Real) = -√two(x) * erfcinv(2x)

"""
    lognormcdf(a, b)

Computes log(normcdf(a, b)), but retaining accuracy.
"""
lognormcdf(a::Real, b::Real) = logerf(a / √two(a), b / √two(b)) - log(two(a + b))

"""
    tnmean(a)

Mean of the standard normal distribution,
truncated to the interval (a, +∞).
"""
tnmean(a::Real) = sqrt(two(a)/π) / erfcx(a/√two(a))

"""
    tnvar(a)

Variance of the standard normal distribution,
truncated to the interval (a, +∞).
WARNING: Fails for very very large values of `a`.
"""
function tnvar(a::Real)
    μ = tnmean(a)
    return one(μ) - (μ - a) * μ
end

"""
    tnstd(a)

Standard deviation of the standard normal distribution,
truncated to the interval (a, +∞).
"""
tnstd(a::Real) = √tnvar(a)

"""
    mills(x::Real)

Mills ratio of the standard normal distribution.
Defined as (1 - cdf(x)) / pdf(x).
"""
mills(x::Real) = erfcx(x / √two(x)) * √(π/two(x))
