"""
    dReLU(; θp, θn, γp, γn)

Double ReLU layer, with separate parameters for positive and negative parts.
"""
struct dReLU{N,A} <: AbstractLayer{N}
    par::A
    function dReLU{N,A}(par::A) where {N,A<:AbstractArray}
        @assert size(par, 1) == 4 # θp, θn, γp, γn
        @assert ndims(par) == N + 1
        return new(par)
    end
end

dReLU(par::AbstractArray) = dReLU{ndims(par) - 1, typeof(par)}(par)

function dReLU(; θp, θn, γp, γn)
    par = vstack((θp, θn, γp, γn))
    return dReLU(par)
end

function dReLU(::Type{T}, sz::Dims) where {T}
    θp = zeros(T, sz)
    θn = zeros(T, sz)
    γp = ones(T, sz)
    γn = ones(T, sz)
    return dReLU(; θp, θn, γp, γn)
end

dReLU(sz::Dims) = dReLU(Float64, sz)
Base.size(layer::dReLU) = size(layer.θp)
Base.size(layer::dReLU, d::Int) = size(layer.θp, d)
Base.length(layer::dReLU) = length(layer.θp)

Base.propertynames(::dReLU) = (:θp, :θn, :γp, :γn)

function Base.getproperty(layer::dReLU, name::Symbol)
    if name === :θp
        return @view getfield(layer, :par)[1, ..]
    elseif name === :θn
        return @view getfield(layer, :par)[2, ..]
    elseif name === :γp
        return @view getfield(layer, :par)[3, ..]
    elseif name === :γn
        return @view getfield(layer, :par)[4, ..]
    else
        return getfield(layer, name)
    end
end

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

std_from_inputs(layer::dReLU, inputs = 0) = sqrt.(var_from_inputs(layer, inputs))

function mean_from_inputs(layer::dReLU, inputs = 0)
    lp = ReLU(; θ =  layer.θp, γ = layer.γp)
    ln = ReLU(; θ = -layer.θn, γ = layer.γn)

    Γp = cgfs(lp,  inputs)
    Γn = cgfs(ln, -inputs)
    Γ = logaddexp.(Γp, Γn)

    pp = exp.(Γp - Γ)
    pn = exp.(Γn - Γ)
    μp = mean_from_inputs(lp,  inputs)
    μn = mean_from_inputs(ln, -inputs)
    return pp .* μp - pn .* μn
end

function var_from_inputs(layer::dReLU, inputs = 0)
    lp = ReLU(; θ =  layer.θp, γ = layer.γp)
    ln = ReLU(; θ = -layer.θn, γ = layer.γn)

    Γp = cgfs(lp,  inputs)
    Γn = cgfs(ln, -inputs)
    Γ = logaddexp.(Γp, Γn)

    pp = exp.(Γp - Γ)
    pn = exp.(Γn - Γ)
    μp, νp = meanvar_from_inputs(lp,  inputs)
    μn, νn = meanvar_from_inputs(ln, -inputs)
    μ = pp .* μp - pn .* μn
    return @. pp * (νp + μp^2) + pn * (νn + μn^2) - μ^2
end

function meanvar_from_inputs(layer::dReLU, inputs = 0)
    lp = ReLU(; θ =  layer.θp, γ = layer.γp)
    ln = ReLU(; θ = -layer.θn, γ = layer.γn)

    Γp = cgfs(lp,  inputs)
    Γn = cgfs(ln, -inputs)
    Γ = logaddexp.(Γp, Γn)

    pp = exp.(Γp - Γ)
    pn = exp.(Γn - Γ)
    μp, νp = meanvar_from_inputs(lp,  inputs)
    μn, νn = meanvar_from_inputs(ln, -inputs)
    μ = pp .* μp - pn .* μn
    ν = @. pp * (νp + μp^2) + pn * (νn + μn^2) - μ^2
    return μ, ν
end

function mean_abs_from_inputs(layer::dReLU, inputs = 0)
    lp = ReLU(; θ =  layer.θp, γ = layer.γp)
    ln = ReLU(; θ = -layer.θn, γ = layer.γn)

    Γp = cgfs(lp,  inputs)
    Γn = cgfs(ln, -inputs)
    Γ = logaddexp.(Γp, Γn)

    pp = exp.(Γp - Γ)
    pn = exp.(Γn - Γ)

    μp =  mean_from_inputs(lp,  inputs)
    μn = -mean_from_inputs(ln, -inputs)
    return pp .* μp - pn .* μn
end

function ∂cgfs(layer::dReLU, inputs = 0)
    lp = ReLU(; θ =  layer.θp, γ = layer.γp)
    ln = ReLU(; θ = -layer.θn, γ = layer.γn)

    Γp = cgfs(lp,  inputs)
    Γn = cgfs(ln, -inputs)
    Γ = logaddexp.(Γp, Γn)

    pp = exp.(Γp - Γ)
    pn = exp.(Γn - Γ)
    μp, νp = meanvar_from_inputs(lp,  inputs)
    μn, νn = meanvar_from_inputs(ln, -inputs)
    μ2p = @. (νp + μp^2) / 2
    μ2n = @. (νn + μn^2) / 2

    ∂θp = +pp .* μp
    ∂θn = -pn .* μn
    ∂γp = -pp .* μ2p .* sign.(layer.γp)
    ∂γn = -pn .* μ2n .* sign.(layer.γn)
    return vstack((∂θp, ∂θn, ∂γp, ∂γn))
end

function moments_from_samples(layer::dReLU, data::AbstractArray; wts = nothing)
    xp = max.(data, 0)
    xn = min.(data, 0)
    xp1 = batchmean(layer, xp; wts)
    xn1 = batchmean(layer, xn; wts)
    xp2 = batchmean(layer, xp.^2; wts)
    xn2 = batchmean(layer, xn.^2; wts)
    return vstack((xp1, xn1, xp2, xn2))
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

function drelu_energy(θp::T, θn::T, γp::S, γn::S, x::Real) where {T<:Real, S<:Real}
    if x ≥ 0
        return gauss_energy(θp, γp, x)
    else
        return gauss_energy(θn, γn, x)
    end
end

function drelu_cgf(θp::Real, θn::Real, γp::Real, γn::Real)
    Γp = relu_cfg( θp, γp)
    Γn = relu_cfg(-θn, γn)
    return logaddexp(Γp, Γn)
end

function drelu_rand(θp::Real, θn::Real, γp::Real, γn::Real)
    return drelu_rand(promote(θp, θn)..., promote(γp, γn)...)
end

function drelu_rand(θp::T, θn::T, γp::S, γn::S) where {T<:Real, S<:Real}
    Γp = relu_cfg(θp, γp)
    Γn = relu_cfg(-θn, γn)
    Γ = logaddexp(Γp, Γn)
    if randexp(typeof(Γ)) ≥ Γ - Γp
        return  relu_rand( θp, γp)
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
