struct xReLU{N, T, A <: AbstractArray{T,N}} <: AbstractLayer{N}
    θ::A
    γ::A
    Δ::A
    ξ::A
    function xReLU(θ::A, γ::A, Δ::A, ξ::A) where {A<:AbstractArray}
        @assert size(θ) == size(γ) == size(Δ) == size(ξ)
        return new{ndims(A), eltype(A), A}(θ, γ, Δ, ξ)
    end
end

function xReLU(::Type{T}, n::Int...) where {T}
    θ = zeros(T, n...)
    γ = ones(T, n...)
    Δ = zeros(T, n...)
    ξ = zeros(T, n...)
    return pReLU(θ, γ, Δ, ξ)
end

xReLU(n::Int...) = xReLU(Float64, n...)

Flux.@functor xReLU

function effective(layer::xReLU, inputs::AbstractTensor; β::Real = true)
    check_size(layer, inputs)
    θ = β * (layer.θ .+ inputs)
    γ = β * broadlike(layer.γ, inputs)
    Δ = β * broadlike(layer.Δ, inputs)
    ξ = broadlike(layer.ξ, inputs)
    return xReLU(promote(θ, γ, Δ, ξ)...)
end

energies(layer::xReLU, x::AbstractArray) = energies(dReLU(layer), x)
free_energies(layer::xReLU) = free_energies(dReLU(layer))
transfer_sample(layer::xReLU) = transfer_sample(dReLU(layer))
transfer_mode(layer::xReLU) = transfer_mode(dReLU(layer))
transfer_mean(layer::xReLU) = transfer_mean(dReLU(layer))
transfer_var(layer::xReLU) = transfer_var(dReLU(layer))
transfer_std(layer::xReLU) = sqrt.(transfer_var(layer))
transfer_mean_abs(layer::xReLU) = transfer_mean_abs(dReLU(layer))

function ∂free_energy(layer::xReLU)
    drelu = dReLU(layer)

    lp = ReLU( drelu.θp, drelu.γp)
    ln = ReLU(-drelu.θn, drelu.γn)

    Fp, Fn = free_energies(lp), free_energies(ln)
    F = -LogExpFunctions.logaddexp.(-Fp, -Fn)
    pp, pn = exp.(F - Fp), exp.(F - Fn)

    μp, μn = transfer_mean(lp), -transfer_mean(ln)
    νp, νn = transfer_var(lp), transfer_var(ln)

    μ2p = @. νp + μp^2
    μ2n = @. νn + μn^2

    η = @. layer.ξ / (1 + abs(layer.ξ))

    ∂θ = @. -(pp * μp + pn * μn)
    ∂γ = @. (pp * μ2p / (1 + η) + pn * μ2n / (1 - η)) / 2
    ∂Δ = @. -(pp * μp / (1 + η) - pn * μn / (1 - η))
    ∂ξ = @. (
        pp * (-layer.γ/2 * μ2p + layer.Δ * μp) / (1 + layer.ξ + abs(layer.ξ))^2 +
        pn * ( layer.γ/2 * μ2n + layer.Δ * μn) / (1 - layer.ξ + abs(layer.ξ))^2
    )
    return (θ = ∂θ, γ = ∂γ, Δ = ∂Δ, ξ = ∂ξ)
end

function ∂energies(layer::xReLU, x::AbstractTensor)
    check_size(layer, x)

    xp = max.(x, 0)
    xn = min.(x, 0)

    η = @. layer.ξ / (1 + abs(layer.ξ))

    ∂θ = @. -x
    ∂γ = @. (xp^2 / (1 + η) + xn^2 / (1 - η)) / 2
    ∂Δ = @. -xp / (1 + η) + xn / (1 - η)
    ∂ξ = @. (
        (-layer.γ/2 * xp^2 + layer.Δ * xp) / (1 + layer.ξ + abs(layer.ξ))^2 +
        ( layer.γ/2 * xn^2 + layer.Δ * xn) / (1 - layer.ξ + abs(layer.ξ))^2
    )
    return (θ = ∂θ, γ = ∂γ, Δ = ∂Δ, ξ = ∂ξ)
end
