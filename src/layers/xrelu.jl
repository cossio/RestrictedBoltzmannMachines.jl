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

function effective(layer::xReLU, inputs::AbstractArray; β::Real = true)
    @assert size(layer) == size(inputs)[1:ndims(layer)]
    θ = β * (layer.θ .+ inputs)
    γ = β * broadlike(layer.γ, inputs)
    Δ = β * broadlike(layer.Δ, inputs)
    ξ = broadlike(layer.ξ, inputs)
    return xReLU(θ, γ, Δ, ξ)
end

energies(layer::xReLU, x::AbstractArray) = energies(dReLU(layer), x)
free_energies(layer::xReLU) = free_energies(dReLU(layer))
transfer_sample(layer::xReLU) = transfer_sample(dReLU(layer))
transfer_mode(layer::xReLU) = transfer_mode(dReLU(layer))
transfer_mean(layer::xReLU) = transfer_mean(dReLU(layer))
transfer_var(layer::xReLU) = transfer_var(dReLU(layer))
transfer_meanvar(layer::xReLU) = transfer_meanvar(dReLU(layer))
transfer_std(layer::xReLU) = sqrt.(transfer_var(layer))
transfer_mean_abs(layer::xReLU) = transfer_mean_abs(dReLU(layer))

function ∂free_energy(layer::xReLU)
    drelu = dReLU(layer)

    lp = ReLU( drelu.θp, drelu.γp)
    ln = ReLU(-drelu.θn, drelu.γn)
    Fp = free_energies(lp)
    Fn = free_energies(ln)
    F = -logaddexp.(-Fp, -Fn)
    pp = exp.(F - Fp)
    pn = exp.(F - Fn)
    μp, νp = transfer_meanvar(lp)
    μn, νn = transfer_meanvar(ln)
    μ2p = @. νp + μp^2
    μ2n = @. νn + μn^2

    η = @. layer.ξ / (1 + abs(layer.ξ))

    ∂θ = @. -(pp * μp - pn * μn)
    ∂γ = @. (pp * μ2p / (1 + η) + pn * μ2n / (1 - η)) / 2
    ∂Δ = @. -(pp * μp / (1 + η) + pn * μn / (1 - η))
    ∂ξ = @. (
        pp * (-layer.γ/2 * μ2p + layer.Δ * μp) / (1 + layer.ξ + abs(layer.ξ))^2 +
        pn * ( layer.γ/2 * μ2n - layer.Δ * μn) / (1 - layer.ξ + abs(layer.ξ))^2
    )
    return (θ = ∂θ, γ = ∂γ, Δ = ∂Δ, ξ = ∂ξ)
end

const xReLUStats = pReLUStats

function ∂energy(layer::xReLU, stats::xReLUStats)
    @assert size(layer) == size(stats)
    η = @. layer.ξ / (1 + abs(layer.ξ))
    ∂γ = @. (stats.xp2 / (1 + η) + stats.xn2 / (1 - η)) / 2
    ∂Δ = @. -stats.xp1 / (1 + η) + stats.xn1 / (1 - η)
    ∂ξ = @. (
        (-layer.γ/2 * stats.xp2 + layer.Δ * stats.xp1) / (1 + layer.ξ + abs(layer.ξ))^2 +
        ( layer.γ/2 * stats.xn2 + layer.Δ * stats.xn1) / (1 - layer.ξ + abs(layer.ξ))^2
    )
    return (θ = -stats.x, γ = ∂γ, Δ = ∂Δ, ξ = ∂ξ)
end

suffstats(layer::xReLU, data::AbstractArray; wts = nothing) = xReLUStats(layer, data; wts)
