"""
    xReLU(; θ, γ = nothing, Δ, ξ)

Extended ReLU layer, like pReLU but with unbounded asymmetry parameter.

The third type parameter of `xReLU{N,A,FixGamma}` is a `Bool` toggle. If `true`, the
scale parameter γ is fixed at 1 and excluded from `par` (which then stores only θ, Δ, ξ).
This removes the gauge invariance between the weights and the hidden unit scales.
[`nsReLU`](@ref) is an alias for this fixed-scale variant, and is what the constructors
return when `γ` is omitted (or `fix_γ = true` is passed).
"""
struct xReLU{N,A,FixGamma} <: AbstractLayer{N}
    par::A
    function xReLU{N,A,FixGamma}(par::A) where {N,A<:AbstractArray,FixGamma}
        FixGamma isa Bool || throw(ArgumentError("FixGamma type parameter must be a Bool"))
        @assert size(par, 1) == (FixGamma ? 3 : 4) # θ, γ, Δ, ξ (θ, Δ, ξ if γ is fixed)
        @assert ndims(par) == N + 1
        return new(par)
    end
end

xReLU(par::AbstractArray; fix_γ::Bool = false) = xReLU{ndims(par) - 1, typeof(par), fix_γ}(par)

function xReLU(; θ, γ = nothing, Δ, ξ)
    if isnothing(γ)
        return xReLU(vstack((θ, Δ, ξ)); fix_γ = true)
    else
        return xReLU(vstack((θ, γ, Δ, ξ)))
    end
end

function xReLU(::Type{T}, sz::Dims; fix_γ::Bool = false) where {T}
    θ = zeros(T, sz)
    Δ = zeros(T, sz)
    ξ = zeros(T, sz)
    if fix_γ
        return xReLU(; θ, Δ, ξ)
    else
        return xReLU(; θ, γ = ones(T, sz), Δ, ξ)
    end
end

xReLU(sz::Dims; kwargs...) = xReLU(Float64, sz; kwargs...)
Base.propertynames(::xReLU) = (:θ, :γ, :Δ, :ξ)

function Base.getproperty(layer::xReLU{N,A,FixGamma}, name::Symbol) where {N,A,FixGamma}
    par = getfield(layer, :par)
    if name === :θ
        return @view par[1, ..]
    elseif name === :γ
        if FixGamma
            return Ones{eltype(par)}(tail(size(par)))
        else
            return @view par[2, ..]
        end
    elseif name === :Δ
        return @view par[FixGamma ? 2 : 3, ..]
    elseif name === :ξ
        return @view par[FixGamma ? 3 : 4, ..]
    else
        return getfield(layer, name)
    end
end

energies(layer::xReLU, x::AbstractArray) = energies(dReLU(layer), x)
cgfs(layer::xReLU, inputs = 0) = cgfs(dReLU(layer), inputs)
sample_from_inputs(layer::xReLU, inputs = 0) = sample_from_inputs(dReLU(layer), inputs)
mode_from_inputs(layer::xReLU, inputs = 0) = mode_from_inputs(dReLU(layer), inputs)
mean_from_inputs(layer::xReLU, inputs = 0) = mean_from_inputs(dReLU(layer), inputs)
var_from_inputs(layer::xReLU, inputs = 0) = var_from_inputs(dReLU(layer), inputs)
meanvar_from_inputs(layer::xReLU, inputs = 0) = meanvar_from_inputs(dReLU(layer), inputs)
std_from_inputs(layer::xReLU, inputs = 0) = sqrt.(var_from_inputs(layer, inputs))
mean_abs_from_inputs(layer::xReLU, inputs = 0) = mean_abs_from_inputs(dReLU(layer), inputs)

function ∂cgfs(layer::xReLU{N,A,FixGamma}, inputs = 0) where {N,A,FixGamma}
    drelu = dReLU(layer)

    lp = ReLU(; θ =  drelu.θp, γ = drelu.γp)
    ln = ReLU(; θ = -drelu.θn, γ = drelu.γn)

    Γp = cgfs(lp,  inputs)
    Γn = cgfs(ln, -inputs)
    Γ = logaddexp.(Γp, Γn)

    pp = exp.(Γp - Γ)
    pn = exp.(Γn - Γ)
    μp, νp = meanvar_from_inputs(lp,  inputs)
    μn, νn = meanvar_from_inputs(ln, -inputs)
    μ2p = @. νp + μp^2
    μ2n = @. νn + μn^2

    η = @. layer.ξ / (1 + abs(layer.ξ))

    ∂θ = @. pp * μp - pn * μn
    ∂Δ = @. (pp * μp / (1 + η) + pn * μn / (1 - η))
    abs_γ = abs.(layer.γ)
    ∂ξ = @. -(
        pp * (-abs_γ/2 * μ2p + layer.Δ * μp) / (1 + layer.ξ + abs(layer.ξ))^2 +
        pn * ( abs_γ/2 * μ2n - layer.Δ * μn) / (1 - layer.ξ + abs(layer.ξ))^2
    )

    if FixGamma
        return vstack((∂θ, ∂Δ, ∂ξ))
    else
        ∂γ = @. -(pp * μ2p / (1 + η) + pn * μ2n / (1 - η)) / 2 * sign(layer.γ)
        return vstack((∂θ, ∂γ, ∂Δ, ∂ξ))
    end
end

function ∂energy_from_moments(layer::xReLU{N,A,FixGamma}, moments::AbstractArray) where {N,A,FixGamma}
    @assert size(moments) == (4, size(layer)...)

    xp1 = moments[1, ..]
    xn1 = moments[2, ..]
    xp2 = moments[3, ..]
    xn2 = moments[4, ..]

    η = @. layer.ξ / (1 + abs(layer.ξ))

    ∂θ = -(xp1 + xn1)
    ∂Δ = @. -xp1 / (1 + η) + xn1 / (1 - η)
    ∂ξ = @. (
        (-abs(layer.γ)/2 * xp2 + layer.Δ * xp1) / (1 + layer.ξ + abs(layer.ξ))^2 +
        ( abs(layer.γ)/2 * xn2 + layer.Δ * xn1) / (1 - layer.ξ + abs(layer.ξ))^2
    )

    if FixGamma
        return vstack((∂θ, ∂Δ, ∂ξ))
    else
        ∂γ = @. sign(layer.γ) * (xp2 / (1 + η) + xn2 / (1 - η)) / 2
        return vstack((∂θ, ∂γ, ∂Δ, ∂ξ))
    end
end
