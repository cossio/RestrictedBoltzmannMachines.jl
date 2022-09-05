struct xReLU{N,A} <: AbstractLayer{N}
    par::A
    function xReLU(par::AbstractArray)
        @assert size(par, 1) == 4 # θ, γ, Δ, ξ
        N = ndims(par) - 1
        return new{N, typeof(par)}(par)
    end
end

function xReLU(; θ, γ, Δ, ξ)
    par = vstack((θ, γ, Δ, ξ))
    return xReLU(par)
end

function xReLU(::Type{T}, sz::Dims) where {T}
    θ = zeros(T, sz)
    γ = ones(T, sz)
    Δ = zeros(T, sz)
    ξ = zeros(T, sz)
    return xReLU(; θ, γ, Δ, ξ)
end

xReLU(sz::Dims) = xReLU(Float64, sz)
Base.propertynames(::xReLU) = (:θ, :γ, :Δ, :ξ)

function Base.getproperty(layer::xReLU, name::Symbol)
    if name === :θ
        return @view getfield(layer, :par)[1, ..]
    elseif name === :γ
        return @view getfield(layer, :par)[2, ..]
    elseif name === :Δ
        return @view getfield(layer, :par)[3, ..]
    elseif name === :ξ
        return @view getfield(layer, :par)[4, ..]
    else
        return getfield(layer, name)
    end
end

energies(layer::xReLU, x::AbstractArray) = energies(dReLU(layer), x)
cfgs(layer::xReLU, inputs = 0) = cfgs(dReLU(layer), inputs)
sample_from_inputs(layer::xReLU, inputs = 0) = sample_from_inputs(dReLU(layer), inputs)
mode_from_inputs(layer::xReLU, inputs = 0) = mode_from_inputs(dReLU(layer), inputs)
mean_from_inputs(layer::xReLU, inputs = 0) = mean_from_inputs(dReLU(layer), inputs)
var_from_inputs(layer::xReLU, inputs = 0) = var_from_inputs(dReLU(layer), inputs)
meanvar_from_inputs(layer::xReLU, inputs = 0) = meanvar_from_inputs(dReLU(layer), inputs)
std_from_inputs(layer::xReLU, inputs = 0) = sqrt.(var_from_inputs(layer, inputs))
mean_abs_from_inputs(layer::xReLU, inputs = 0) = mean_abs_from_inputs(dReLU(layer), inputs)

function ∂cfgs(layer::xReLU, inputs = 0)
    drelu = dReLU(layer)

    lp = ReLU(; θ =  drelu.θp, γ = drelu.γp)
    ln = ReLU(; θ = -drelu.θn, γ = drelu.γn)

    Fp = cfgs(lp,  inputs)
    Fn = cfgs(ln, -inputs)
    F = -logaddexp.(-Fp, -Fn)

    pp = exp.(F - Fp)
    pn = exp.(F - Fn)
    μp, νp = meanvar_from_inputs(lp,  inputs)
    μn, νn = meanvar_from_inputs(ln, -inputs)
    μ2p = @. νp + μp^2
    μ2n = @. νn + μn^2

    η = @. layer.ξ / (1 + abs(layer.ξ))

    ∂θ = @. -(pp * μp - pn * μn)
    ∂γ = @. (pp * μ2p / (1 + η) + pn * μ2n / (1 - η)) / 2
    ∂γ .*= sign.(layer.γ)
    ∂Δ = @. -(pp * μp / (1 + η) + pn * μn / (1 - η))
    abs_γ = abs.(layer.γ)
    ∂ξ = @. (
        pp * (-abs_γ/2 * μ2p + layer.Δ * μp) / (1 + layer.ξ + abs(layer.ξ))^2 +
        pn * ( abs_γ/2 * μ2n - layer.Δ * μn) / (1 - layer.ξ + abs(layer.ξ))^2
    )

    return vstack((∂θ, ∂γ, ∂Δ, ∂ξ))
end

function ∂energy_from_moments(layer::xReLU, moments::AbstractArray)
    @assert size(layer.par) == size(moments)

    xp1 = moments[1, ..]
    xn1 = moments[2, ..]
    xp2 = moments[3, ..]
    xn2 = moments[4, ..]

    η = @. layer.ξ / (1 + abs(layer.ξ))

    ∂θ = -(xp1 + xn1)
    ∂γ = @. sign(layer.γ) * (xp2 / (1 + η) + xn2 / (1 - η)) / 2
    ∂Δ = @. -xp1 / (1 + η) + xn1 / (1 - η)
    ∂ξ = @. (
        (-abs(layer.γ)/2 * xp2 + layer.Δ * xp1) / (1 + layer.ξ + abs(layer.ξ))^2 +
        ( abs(layer.γ)/2 * xn2 + layer.Δ * xn1) / (1 - layer.ξ + abs(layer.ξ))^2
    )

    return vstack((∂θ, ∂γ, ∂Δ, ∂ξ))
end
