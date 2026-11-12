@doc raw"""
    dReLU(; θp, θn, γp, γn)

Double ReLU layer, with separate parameters for positive and negative parts.
The energy of a layer with units ``h_\mu`` is ``E = \sum_\mu U(h_\mu)``, with the
unit potential:

```math
U(h) = \begin{cases}
    \frac{|\gamma^+|}{2} h^2 - \theta^+ h & h \ge 0 \\[4pt]
    \frac{|\gamma^-|}{2} h^2 - \theta^- h & h < 0
\end{cases}
```

where ``\theta^+, \theta^-, \gamma^+, \gamma^-`` are the entries of
`θp`, `θn`, `γp`, `γn` for the corresponding unit, and ``h`` takes values in
``\mathbb{R}``.
"""
@declare_layer dReLU (θp = zeros, θn = zeros, γp = ones, γn = ones)

function energies(layer::dReLU, x::AbstractArray)
    @assert size(layer) == size(x)[1:ndims(layer)]
    return drelu_energy.(layer.θp, layer.θn, layer.γp, layer.γn, x)
end

cgfs(layer::dReLU, inputs::AbstractArray = Falses(size(layer))) = drelu_cgf.(layer.θp .+ inputs, layer.θn .+ inputs, layer.γp, layer.γn)
sample_from_inputs(layer::dReLU, inputs::AbstractArray = Falses(size(layer))) = drelu_rand.(layer.θp .+ inputs, layer.θn .+ inputs, layer.γp, layer.γn)
mode_from_inputs(layer::dReLU, inputs::AbstractArray = Falses(size(layer))) = drelu_mode.(layer.θp .+ inputs, layer.θn .+ inputs, layer.γp, layer.γn)

mean_from_inputs(layer::dReLU, inputs::AbstractArray = Falses(size(layer))) = first.(drelu_meanvar.(layer.θp .+ inputs, layer.θn .+ inputs, layer.γp, layer.γn))
var_from_inputs(layer::dReLU, inputs::AbstractArray = Falses(size(layer))) = last.(drelu_meanvar.(layer.θp .+ inputs, layer.θn .+ inputs, layer.γp, layer.γn))
meanvar_from_inputs(layer::dReLU, inputs::AbstractArray = Falses(size(layer))) = _unzip(drelu_meanvar.(layer.θp .+ inputs, layer.θn .+ inputs, layer.γp, layer.γn))
mean_abs_from_inputs(layer::dReLU, inputs::AbstractArray = Falses(size(layer))) = drelu_mean_abs.(layer.θp .+ inputs, layer.θn .+ inputs, layer.γp, layer.γn)
moments_from_inputs(layer::dReLU, inputs::AbstractArray = Falses(size(layer))) = _stack_tuples(drelu_moments.(layer.θp .+ inputs, layer.θn .+ inputs, layer.γp, layer.γn))

function ∂energy_from_moments(layer::dReLU, moments::AbstractArray)
    _check_moments(layer, moments)
    ∂θp = @views -moments[1, ..]
    ∂θn = @views -moments[2, ..]
    ∂γp = @views sign.(layer.γp) .* moments[3, ..] / 2
    ∂γn = @views sign.(layer.γn) .* moments[4, ..] / 2
    return stack([∂θp, ∂θn, ∂γp, ∂γn]; dims = 1)
end

function drelu_energy(θp::Real, θn::Real, γp::Real, γn::Real, x::Real)
    Ep, En = promote(gauss_energy(θp, γp, x), gauss_energy(θn, γn, x))
    return ifelse(x ≥ 0, Ep, En)
end

function drelu_cgf(θp::Real, θn::Real, γp::Real, γn::Real)
    Γp = relu_cgf(θp, γp)
    Γn = relu_cgf(-θn, γn)
    return logaddexp(Γp, Γn)
end

#=
A dReLU unit is a two-sided mixture of truncated Gaussians: a positive-side ReLU with
parameters (θp, γp) and a mirrored negative-side ReLU with parameters (-θn, γn).
Returns the mixture weights `pp`, `pn` and the mean and variance `μp`, `νp`, `μn`, `νn`
of each side (with the negative side mirrored to positive values), from which the
statistics below follow.
=#
function _drelu_mixture(θp::Real, θn::Real, γp::Real, γn::Real)
    Δ = _drelu_logodds(θp, θn, γp, γn)
    μp, νp = relu_meanvar(θp, γp)
    μn, νn = relu_meanvar(-θn, γn)
    # logistic functions of the same log-odds: the weights sum to one
    return (; pp = logistic(Δ), pn = logistic(-Δ), μp, μn, νp, νn)
end

#=
Log-odds of the positive side of a dReLU unit: the difference of the cgfs of the ReLUs of its
two sides. A side with θ > 0 has a well of depth θ^2 / 2|γ|, which dominates its cgf when
deep. With deep wells on both sides, the difference of the cgfs as computed by `relu_cgf`
would carry rounding errors proportional to the depths, and so would the weights of the
sides. Here the depths are subtracted first, together with their rounding errors, and the
remainders of the cgfs are of order one.
=#
function _drelu_logodds(θp::Real, θn::Real, γp::Real, γn::Real)
    dp, ep, rp = _relu_cgf_parts(θp, γp)
    dn, en, rn = _relu_cgf_parts(-θn, γn)
    return (dp - dn) + ((ep - en) + (rp - rn))
end

#=
Splits `relu_cgf(θ, γ)` into the sum of the depth `d` of the well (zero for θ ≤ 0), the
rounding error `e` of `d`, and a remainder `r`, using that logerfcx(z) = z^2 + logerfc(z).
=#
function _relu_cgf_parts(θ::Real, γ::Real)
    abs_2γ = 2abs(γ)
    t = max(θ, zero(θ))
    t2 = t * t
    d = t2 / abs_2γ
    # (t^2 - d * 2|γ|) / 2|γ|: the fused operations give the errors of the two products
    e = (fma(-d, abs_2γ, t2) + fma(t, t, -t2)) / abs_2γ
    z = -θ / √abs_2γ
    r = (θ > 0 ? logerfc(z) : logerfcx(z)) - log(abs_2γ / π) / 2
    return d, e, r
end

function drelu_meanvar(θp::Real, θn::Real, γp::Real, γn::Real)
    (; pp, pn, μp, μn, νp, νn) = _drelu_mixture(θp, θn, γp, γn)
    μ = pp * μp - pn * μn
    # law of total variance: within the sides, plus between their means μp and -μn
    ν = pp * νp + pn * νn + pp * pn * (μp + μn)^2
    return μ, ν
end

function drelu_mean_abs(θp::Real, θn::Real, γp::Real, γn::Real)
    (; pp, pn, μp, μn) = _drelu_mixture(θp, θn, γp, γn)
    return pp * μp + pn * μn
end

# the four moment slots `<xp>`, `<xn>`, `<xp^2>`, `<xn^2>`
function drelu_moments(θp::Real, θn::Real, γp::Real, γn::Real)
    (; pp, pn, μp, μn, νp, νn) = _drelu_mixture(θp, θn, γp, γn)
    # the negative side is mirrored back to x = -y ≤ 0
    return pp * μp, -pn * μn, pp * (νp + μp^2), pn * (νn + μn^2)
end

function drelu_rand(θp::Real, θn::Real, γp::Real, γn::Real)
    return drelu_rand(promote(θp, θn)..., promote(γp, γn)...)
end

function drelu_rand(θp::T, θn::T, γp::S, γn::S) where {T <: Real, S <: Real}
    Δ = _drelu_logodds(θp, θn, γp, γn)
    # positive side with probability logistic(Δ) = exp(-log1pexp(-Δ))
    if randexp(typeof(Δ)) ≥ log1pexp(-Δ)
        return relu_rand(θp, γp)
    else
        return -relu_rand(-θn, γn)
    end
end

function drelu_mode(θp::Real, θn::Real, γp::Real, γn::Real)
    T = promote_type(typeof(θp / abs(γp)), typeof(θn / abs(γn)))
    if θp ≤ 0 ≤ θn
        return zero(T)
    elseif θn ≤ 0 ≤ θp && θp^2 / abs(γp) ≥ θn^2 / abs(γn) || θp ≥ 0 && θn ≥ 0
        return convert(T, θp / abs(γp))
    elseif θn ≤ 0 ≤ θp && θp^2 / abs(γp) ≤ θn^2 / abs(γn) || θp ≤ 0 && θn ≤ 0
        return convert(T, θn / abs(γn))
    else
        return convert(T, NaN)
    end
end
