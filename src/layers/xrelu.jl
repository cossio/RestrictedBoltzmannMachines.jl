struct xReLU{N,Aθ,Aγ,AΔ,Aξ} <: AbstractLayer{N}
    θ::Aθ
    γ::Aγ
    Δ::AΔ
    ξ::Aξ
    function xReLU(θ::AbstractArray, γ::AbstractArray, Δ::AbstractArray, ξ::AbstractArray)
        @assert size(θ) == size(γ) == size(Δ) == size(ξ)
        return new{ndims(θ), typeof(θ), typeof(γ), typeof(Δ), typeof(ξ)}(θ, γ, Δ, ξ)
    end
end

function xReLU(::Type{T}, n::Int...) where {T}
    θ = zeros(T, n)
    γ = ones(T, n)
    Δ = zeros(T, n)
    ξ = zeros(T, n)
    return xReLU(θ, γ, Δ, ξ)
end

xReLU(n::Int...) = xReLU(Float64, n...)

Base.repeat(l::xReLU, n::Int...) = xReLU(
    repeat(l.θ, n...), repeat(l.γ, n...), repeat(l.Δ, n...), repeat(l.ξ, n...)
)

energies(layer::xReLU, x::AbstractArray) = energies(dReLU(layer), x)
free_energies(layer::xReLU, inputs::Union{Real,AbstractArray} = 0) = free_energies(dReLU(layer), inputs)
transfer_sample(layer::xReLU, inputs::Union{Real,AbstractArray} = 0) = transfer_sample(dReLU(layer), inputs)
transfer_mode(layer::xReLU, inputs::Union{Real,AbstractArray} = 0) = transfer_mode(dReLU(layer), inputs)
transfer_mean(layer::xReLU, inputs::Union{Real,AbstractArray} = 0) = transfer_mean(dReLU(layer), inputs)
var_from_inputs(layer::xReLU, inputs::Union{Real,AbstractArray} = 0) = var_from_inputs(dReLU(layer), inputs)
meanvar_from_inputs(layer::xReLU, inputs::Union{Real,AbstractArray} = 0) = meanvar_from_inputs(dReLU(layer), inputs)
std_from_inputs(layer::xReLU, inputs::Union{Real,AbstractArray} = 0) = sqrt.(var_from_inputs(layer, inputs))
transfer_mean_abs(layer::xReLU, inputs::Union{Real,AbstractArray} = 0) = transfer_mean_abs(dReLU(layer), inputs)

function ∂free_energies(layer::xReLU, inputs::Union{Real,AbstractArray} = 0)
    drelu = dReLU(layer)

    lp = ReLU( drelu.θp, drelu.γp)
    ln = ReLU(-drelu.θn, drelu.γn)
    Fp = free_energies(lp,  inputs)
    Fn = free_energies(ln, -inputs)
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
    return (θ = ∂θ, γ = ∂γ, Δ = ∂Δ, ξ = ∂ξ)
end

const xReLUStats = pReLUStats

function ∂energy(layer::xReLU, stats::xReLUStats)
    @assert size(layer) == size(stats)
    η = @. layer.ξ / (1 + abs(layer.ξ))
    ∂γ = @. (stats.xp2 / (1 + η) + stats.xn2 / (1 - η)) / 2
    ∂γ .*= sign.(layer.γ)
    ∂Δ = @. -stats.xp1 / (1 + η) + stats.xn1 / (1 - η)
    abs_γ = abs.(layer.γ)
    ∂ξ = @. (
        (-abs_γ/2 * stats.xp2 + layer.Δ * stats.xp1) / (1 + layer.ξ + abs(layer.ξ))^2 +
        ( abs_γ/2 * stats.xn2 + layer.Δ * stats.xn1) / (1 - layer.ξ + abs(layer.ξ))^2
    )
    return (θ = -stats.x, γ = ∂γ, Δ = ∂Δ, ξ = ∂ξ)
end

suffstats(layer::xReLU, data::AbstractArray; wts = nothing) = xReLUStats(layer, data; wts)
