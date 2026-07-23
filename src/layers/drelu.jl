"""
    dReLU(; θp, θn, γp, γn)

Double ReLU layer, with separate parameters for positive and negative parts.
"""
@declare_layer dReLU (θp = zeros, θn = zeros, γp = ones, γn = ones)

function energies(layer::dReLU, x::AbstractArray)
    @assert size(layer) == size(x)[1:ndims(layer)]
    return drelu_energy.(layer.θp, layer.θn, layer.γp, layer.γn, x)
end

function cgfs(layer::dReLU, inputs = 0)
    θp = layer.θp .+ inputs
    θn = layer.θn .+ inputs
    return drelu_cgf.(θp, θn, layer.γp, layer.γn)
end

function sample_from_inputs(layer::dReLU, inputs = 0)
    θp = layer.θp .+ inputs
    θn = layer.θn .+ inputs
    return drelu_rand.(θp, θn, layer.γp, layer.γn)
end

function mode_from_inputs(layer::dReLU, inputs = 0)
    θp = layer.θp .+ inputs
    θn = layer.θn .+ inputs
    return drelu_mode.(θp, θn, layer.γp, layer.γn)
end

#=
A dReLU unit is a two-sided mixture of truncated Gaussians: a positive-side ReLU with
parameters (θp, γp) and a mirrored negative-side ReLU with parameters (-θn, γn).
Returns the mixture weights `pp`, `pn` and the mean and variance `μp`, `νp`, `μn`, `νn`
of each side (with the negative side mirrored to positive values). This is the shared
preamble of the dReLU statistics, and of the pReLU / xReLU gradients.
=#
function _drelu_mixture_moments(layer::dReLU, inputs)
    lp = ReLU(; θ = layer.θp, γ = layer.γp)
    ln = ReLU(; θ = -layer.θn, γ = layer.γn)

    Γp = cgfs(lp, inputs)
    Γn = cgfs(ln, -inputs)
    Γ = logaddexp.(Γp, Γn)

    pp = exp.(Γp - Γ)
    pn = exp.(Γn - Γ)
    μp, νp = meanvar_from_inputs(lp, inputs)
    μn, νn = meanvar_from_inputs(ln, -inputs)
    return (; pp, pn, μp, μn, νp, νn)
end

mean_from_inputs(layer::dReLU, inputs = 0) = first(meanvar_from_inputs(layer, inputs))
var_from_inputs(layer::dReLU, inputs = 0) = last(meanvar_from_inputs(layer, inputs))

function meanvar_from_inputs(layer::dReLU, inputs = 0)
    (; pp, pn, μp, μn, νp, νn) = _drelu_mixture_moments(layer, inputs)
    μ = pp .* μp - pn .* μn
    ν = @. pp * (νp + μp^2) + pn * (νn + μn^2) - μ^2
    return μ, ν
end

function mean_abs_from_inputs(layer::dReLU, inputs = 0)
    (; pp, pn, μp, μn) = _drelu_mixture_moments(layer, inputs)
    return pp .* μp + pn .* μn
end

function ∂cgfs(layer::dReLU, inputs = 0)
    (; pp, pn, μp, μn, νp, νn) = _drelu_mixture_moments(layer, inputs)
    μ2p = @. (νp + μp^2) / 2
    μ2n = @. (νn + μn^2) / 2

    ∂θp = +pp .* μp
    ∂θn = -pn .* μn
    ∂γp = -pp .* μ2p .* sign.(layer.γp)
    ∂γn = -pn .* μ2n .* sign.(layer.γn)
    return vstack((∂θp, ∂θn, ∂γp, ∂γn))
end

function ∂energy_from_moments(layer::dReLU, moments::AbstractArray)
    @assert size(layer.par) == size(moments)
    ∂θp = -moments[1, ..]
    ∂θn = -moments[2, ..]
    ∂γp = sign.(layer.γp) .* moments[3, ..] / 2
    ∂γn = sign.(layer.γn) .* moments[4, ..] / 2
    return vstack((∂θp, ∂θn, ∂γp, ∂γn))
end

function drelu_energy(θp::Real, θn::Real, γp::Real, γn::Real, x::Real)
    return drelu_energy(promote(θp, θn)..., promote(γp, γn)..., x)
end

function drelu_energy(θp::T, θn::T, γp::S, γn::S, x::Real) where {T <: Real, S <: Real}
    if x ≥ 0
        return gauss_energy(θp, γp, x)
    else
        return gauss_energy(θn, γn, x)
    end
end

function drelu_cgf(θp::Real, θn::Real, γp::Real, γn::Real)
    Γp = relu_cfg(θp, γp)
    Γn = relu_cfg(-θn, γn)
    return logaddexp(Γp, Γn)
end

function drelu_rand(θp::Real, θn::Real, γp::Real, γn::Real)
    return drelu_rand(promote(θp, θn)..., promote(γp, γn)...)
end

function drelu_rand(θp::T, θn::T, γp::S, γn::S) where {T <: Real, S <: Real}
    Γp = relu_cfg(θp, γp)
    Γn = relu_cfg(-θn, γn)
    Γ = logaddexp(Γp, Γn)
    if randexp(typeof(Γ)) ≥ Γ - Γp
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
