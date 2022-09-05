struct pReLU{N,A} <: AbstractLayer{N}
    par::A
    function pReLU(par::AbstractArray)
        @assert size(par, 1) == 4 # θ, γ, Δ, η
        N = ndims(par) - 1
        return new{N, typeof(par)}(par)
    end
end

function pReLU(; θ, γ, Δ, η)
    par = vstack((θ, γ, Δ, η))
    return pReLU(par)
end

function pReLU(::Type{T}, sz::Dims) where {T}
    θ = zeros(T, sz)
    γ = ones(T, sz)
    Δ = zeros(T, sz)
    η = zeros(T, sz)
    return pReLU(; θ, γ, Δ, η)
end

pReLU(sz::Dims) = pReLU(Float64, sz)
Base.propertynames(::pReLU) = (:θ, :γ, :Δ, :η)

function Base.getproperty(layer::pReLU, name::Symbol)
    if name === :θ
        return @view getfield(layer, :par)[1, ..]
    elseif name === :γ
        return @view getfield(layer, :par)[2, ..]
    elseif name === :Δ
        return @view getfield(layer, :par)[3, ..]
    elseif name === :η
        return @view getfield(layer, :par)[4, ..]
    else
        return getfield(layer, name)
    end
end

energies(layer::pReLU, x::AbstractArray) = energies(dReLU(layer), x)
cfgs(layer::pReLU, inputs = 0) = cfgs(dReLU(layer), inputs)
sample_from_inputs(layer::pReLU, inputs = 0) = sample_from_inputs(dReLU(layer), inputs)
mean_from_inputs(layer::pReLU, inputs = 0) = mean_from_inputs(dReLU(layer), inputs)
var_from_inputs(layer::pReLU, inputs = 0) = var_from_inputs(dReLU(layer), inputs)
meanvar_from_inputs(layer::pReLU, inputs = 0) = meanvar_from_inputs(dReLU(layer), inputs)
mode_from_inputs(layer::pReLU, inputs = 0) = mode_from_inputs(dReLU(layer), inputs)
mean_abs_from_inputs(layer::pReLU, inputs = 0) = mean_abs_from_inputs(dReLU(layer), inputs)
std_from_inputs(layer::pReLU, inputs = 0) = sqrt.(var_from_inputs(layer, inputs))

function ∂cfgs(layer::pReLU, inputs = 0)
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
    μ2p = @. (νp + μp^2) / 2
    μ2n = @. (νn + μn^2) / 2

    ∂θ = @. -(pp * μp - pn * μn)
    ∂γ = @. sign(layer.γ) * (pp * μ2p / (1 + layer.η) + pn * μ2n / (1 - layer.η))
    ∂Δ = @. -pp * μp / (1 + layer.η) - pn * μn / (1 - layer.η)
    ∂η = @. (
        pp * (-abs(layer.γ) * μ2p + layer.Δ * μp) / (1 + layer.η)^2 +
        pn * ( abs(layer.γ) * μ2n - layer.Δ * μn) / (1 - layer.η)^2
    )

    return vstack((∂θ, ∂γ, ∂Δ, ∂η))
end

function ∂energy_from_moments(layer::pReLU, moments::AbstractArray)
    @assert size(layer.par) == size(moments)

    xp1 = @view moments[1, ..]
    xn1 = @view moments[2, ..]
    xp2 = @view moments[3, ..]
    xn2 = @view moments[4, ..]

    ∂θ = -(xp1 + xn1)
    ∂γ = @. sign(layer.γ) * (xp2 / (1 + layer.η) + xn2 / (1 - layer.η)) / 2
    ∂Δ = @. -(xp1 / (1 + layer.η) - xn1 / (1 - layer.η))
    ∂η = @. (
        (-abs(layer.γ) * xp2 / 2 + layer.Δ * xp1) / (1 + layer.η)^2 +
        ( abs(layer.γ) * xn2 / 2 + layer.Δ * xn1) / (1 - layer.η)^2
    )

    return vstack((∂θ, ∂γ, ∂Δ, ∂η))
end
