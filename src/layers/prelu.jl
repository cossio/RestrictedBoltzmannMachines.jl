struct pReLU{N,Aθ,Aγ,AΔ,Aη} <: AbstractLayer{N}
    θ::Aθ
    γ::Aγ
    Δ::AΔ
    η::Aη
    function pReLU(θ::AbstractArray, γ::AbstractArray, Δ::AbstractArray, η::AbstractArray)
        @assert size(θ) == size(γ) == size(Δ) == size(η)
        return new{ndims(θ), typeof(θ), typeof(γ), typeof(Δ), typeof(η)}(θ, γ, Δ, η)
    end
end

function pReLU(::Type{T}, n::Int...) where {T}
    θ = zeros(T, n...)
    γ = ones(T, n...)
    Δ = zeros(T, n...)
    η = zeros(T, n...)
    return pReLU(θ, γ, Δ, η)
end

pReLU(n::Int...) = pReLU(Float64, n...)

Base.repeat(l::pReLU, n::Int...) = pReLU(
    repeat(l.θ, n...), repeat(l.γ, n...), repeat(l.Δ, n...), repeat(l.η, n...)
)

function effective(layer::pReLU, inputs::AbstractArray; β::Real = true)
    θ = β * (layer.θ .+ inputs)
    γ = β * broadlike(layer.γ, inputs)
    Δ = β * broadlike(layer.Δ, inputs)
    η = broadlike(layer.η, inputs)
    return pReLU(θ, γ, Δ, η)
end

energies(layer::pReLU, x::AbstractArray) = energies(dReLU(layer), x)
free_energies(layer::pReLU) = free_energies(dReLU(layer))
transfer_sample(layer::pReLU) = transfer_sample(dReLU(layer))
transfer_mean(layer::pReLU) = transfer_mean(dReLU(layer))
transfer_var(layer::pReLU) = transfer_var(dReLU(layer))
transfer_meanvar(layer::pReLU) = transfer_meanvar(dReLU(layer))
transfer_mode(layer::pReLU) = transfer_mode(dReLU(layer))
transfer_mean_abs(layer::pReLU) = transfer_mean_abs(dReLU(layer))
transfer_std(layer::pReLU) = sqrt.(transfer_var(layer))

function ∂free_energy(layer::pReLU)
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
    μ2p = @. (νp + μp^2) / 2
    μ2n = @. (νn + μn^2) / 2

    ∂θ = @. -(pp * μp - pn * μn)
    ∂γ = @. pp * μ2p / (1 + layer.η) + pn * μ2n / (1 - layer.η)
    ∂Δ = @. -pp * μp / (1 + layer.η) - pn * μn / (1 - layer.η)
    ∂η = @. (
        pp * (-layer.γ * μ2p + layer.Δ * μp) / (1 + layer.η)^2 +
        pn * ( layer.γ * μ2n - layer.Δ * μn) / (1 - layer.η)^2
    )
    return (θ = ∂θ, γ = ∂γ, Δ = ∂Δ, η = ∂η)
end

struct pReLUStats{A<:AbstractArray}
    x::A; xp1::A; xn1::A; xp2::A; xn2::A;
    function pReLUStats(layer::AbstractLayer, data::AbstractArray; wts = nothing)
        xp = max.(data, 0)
        xn = min.(data, 0)
        x = batchmean(layer, data; wts)
        xp1 = batchmean(layer, xp; wts)
        xn1 = batchmean(layer, xn; wts)
        xp2 = batchmean(layer, xp.^2; wts)
        xn2 = batchmean(layer, xn.^2; wts)
        return new{typeof(x)}(x, xp1, xn1, xp2, xn2)
    end
end

Base.size(stats::pReLUStats) = size(stats.x)

function ∂energy(layer::pReLU, stats::pReLUStats)
    @assert size(layer) == size(stats)
    ∂γ = @. (stats.xp2 / (1 + layer.η) + stats.xn2 / (1 - layer.η)) / 2
    ∂Δ = @. -(stats.xp1 / (1 + layer.η) - stats.xn1 / (1 - layer.η))
    ∂η = @. (
        (-layer.γ * stats.xp2 / 2 + layer.Δ * stats.xp1) / (1 + layer.η)^2 +
        ( layer.γ * stats.xn2 / 2 + layer.Δ * stats.xn1) / (1 - layer.η)^2
    )
    return (θ = -stats.x, γ = ∂γ, Δ = ∂Δ, η = ∂η)
end

suffstats(layer::pReLU, data::AbstractArray; wts = nothing) = pReLUStats(layer, data; wts)
