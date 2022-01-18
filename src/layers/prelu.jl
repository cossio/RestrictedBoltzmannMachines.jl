struct pReLU{N, T, A <: AbstractArray{T,N}} <: AbstractLayer{N}
    θ::A
    γ::A
    Δ::A
    η::A
    function pReLU(θ::A, γ::A, Δ::A, η::A) where {A<:AbstractArray}
        @assert size(θ) == size(γ) == size(Δ) == size(η)
        return new{ndims(A), eltype(A), A}(θ, γ, Δ, η)
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

Flux.@functor pReLU

function effective(layer::pReLU, inputs::AbstractArray; β::Real = true)
    θ = β * (layer.θ .+ inputs)
    γ = β * broadlike(layer.γ, inputs)
    Δ = β * broadlike(layer.Δ, inputs)
    η = broadlike(layer.η, inputs)
    return pReLU(promote(θ, γ, Δ, η)...)
end

function energies(layer::pReLU, x::AbstractArray)
    return energies(dReLU(layer), x)
end

function free_energies(layer::pReLU)
    return free_energies(dReLU(layer))
end

function transfer_sample(layer::pReLU)
    return transfer_sample(dReLU(layer))
end

function transfer_mean(layer::pReLU)
    return transfer_mean(dReLU(layer))
end

function transfer_var(layer::pReLU)
    return transfer_var(dReLU(layer))
end

function transfer_mode(layer::pReLU)
    return transfer_mode(dReLU(layer))
end

transfer_mean_abs(layer::pReLU) = transfer_mean_abs(dReLU(layer))
transfer_std(layer::pReLU) = sqrt.(transfer_var(layer))

function ∂free_energy(layer::pReLU)
    drelu = dReLU(layer)

    lp = ReLU( drelu.θp, drelu.γp)
    ln = ReLU(-drelu.θn, drelu.γn)

    Fp, Fn = free_energies(lp), free_energies(ln)
    F = -LogExpFunctions.logaddexp.(-Fp, -Fn)
    pp, pn = exp.(F - Fp), exp.(F - Fn)

    μp, μn = transfer_mean(lp), -transfer_mean(ln)
    νp, νn = transfer_var(lp), transfer_var(ln)

    μ2p = @. (νp + μp^2) / 2
    μ2n = @. (νn + μn^2) / 2

    ∂θ = @. -(pp * μp + pn * μn)
    ∂γ = @. pp * μ2p / (1 + layer.η) + pn * μ2n / (1 - layer.η)
    ∂Δ = @. -pp * μp / (1 + layer.η) + pn * μn / (1 - layer.η)
    ∂η = @. (
        pp * (-layer.γ * μ2p + layer.Δ * μp) / (1 + layer.η)^2 +
        pn * ( layer.γ * μ2n + layer.Δ * μn) / (1 - layer.η)^2
    )
    return (θ = ∂θ, γ = ∂γ, Δ = ∂Δ, η = ∂η)
end

function ∂energy(layer::pReLU; x, xp, xn, xp2, xn2)
    for ξ in (x, xp, xn, xp2, xn2)
        @assert size(ξ::AbstractArray) == size(layer)
    end

    ∂θ = @. -x
    ∂γ = @. (xp2 / (1 + layer.η) + xn2 / (1 - layer.η)) / 2
    ∂Δ = @. -(xp / (1 + layer.η) - xn / (1 - layer.η))
    ∂η = @. (
        (-layer.γ * xp2 / 2 + layer.Δ * xp) / (1 + layer.η)^2 +
        ( layer.γ * xn2 / 2 + layer.Δ * xn) / (1 - layer.η)^2
    )

    return (θ = ∂θ, γ = ∂γ, Δ = ∂Δ, η = ∂η)
end

function sufficient_statistics(layer::pReLU, x::AbstractArray; wts = nothing)
    @assert size(layer) == size(x)[1:ndims(layer)]

    xp = max.(x, 0)
    xn = min.(x, 0)

    μ = batchmean(layer, x; wts)
    μp = batchmean(layer, xp; wts)
    μn = batchmean(layer, xn; wts)
    μp2 = batchmean(layer, xp.^2; wts)
    μn2 = batchmean(layer, xn.^2; wts)

    return (x = μ, xp = μp, xn = μn, xp2 = μp2, xn2 = μn2)
end
