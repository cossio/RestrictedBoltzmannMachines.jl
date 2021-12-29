struct xReLU{A<:AbstractArray}
    θ::A
    γ::A
    Δ::A
    ξ::A
    function xReLU(θ::A, γ::A, Δ::A, ξ::A) where {A<:AbstractArray}
        @assert size(θ) == size(γ) == size(Δ) == size(ξ)
        return new{A}(θ, γ, Δ, ξ)
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

function energies(layer::xReLU, x::AbstractArray)
    energies(dReLU(layer), x)
end

cgfs(layer::xReLU) = cgfs(dReLU(layer))

function transfer_sample(layer::xReLU)
    return transfer_sample(dReLU(layer))
end

transfer_mode(layer::xReLU) = transfer_mode(dReLU(layer))
transfer_mean(layer::xReLU) = transfer_mean(dReLU(layer))
transfer_var(layer::xReLU) = transfer_var(dReLU(layer))
transfer_mean_abs(layer::xReLU) = transfer_mean_abs(dReLU(layer))

function conjugates(layer::xReLU)
    drelu = dReLU(layer)

    lp = ReLU( drelu.θp, drelu.γp)
    ln = ReLU(-drelu.θn, drelu.γn)

    Γp, Γn = cgfs(lp), cgfs(ln)
    Γ = LogExpFunctions.logaddexp.(Γp, Γn)
    pp, pn = exp.(Γp - Γ), exp.(Γn - Γ)

    μp, μn = transfer_mean(lp), -transfer_mean(ln)
    νp, νn = transfer_var(lp), transfer_var(ln)

    μ2p = @. νp + μp^2
    μ2n = @. νn + μn^2

    η = @. layer.ξ / (1 + abs(layer.ξ))

    ∂θ = @. pp * μp + pn * μn
    ∂γ = @. -(pp * μ2p / (1 + η) + pn * μ2n / (1 - η)) / 2
    ∂Δ = @. pp * μp / (1 + η) - pn * μn / (1 - η)
    ∂ξ = @. (
        pp * ( layer.γ/2 * μ2p - layer.Δ * μp) / (1 + layer.ξ + abs(layer.ξ))^2 +
        pn * (-layer.γ/2 * μ2n - layer.Δ * μn) / (1 - layer.ξ + abs(layer.ξ))^2
    )

    return (θ = ∂θ, γ = ∂γ, Δ = ∂Δ, ξ = ∂ξ)
end

function conjugates_empirical(layer::xReLU, samples::AbstractArray)
    @assert size(samples) == (size(layer)..., size(samples)[end])

    xp = max.(samples, 0)
    xn = min.(samples, 0)

    μp = mean_(xp; dims=ndims(samples))
    μn = mean_(xn; dims=ndims(samples))

    μ2p = mean_(xp.^2; dims=ndims(samples))
    μ2n = mean_(xn.^2; dims=ndims(samples))

    η = @. layer.ξ / (1 + abs(layer.ξ))

    ∂θ = @. μp + μn
    ∂γ = @. -(μ2p / (1 + η) + μ2n / (1 - η)) / 2
    ∂Δ = @. μp / (1 + η) - μn / (1 - η)
    ∂ξ = @. (
        ( layer.γ/2 * μ2p - layer.Δ * μp) / (1 + layer.ξ + abs(layer.ξ))^2 +
        (-layer.γ/2 * μ2n - layer.Δ * μn) / (1 - layer.ξ + abs(layer.ξ))^2
    )

    return (θ = ∂θ, γ = ∂γ, Δ = ∂Δ, ξ = ∂ξ)
end

function effective(layer::xReLU, inputs, β::Real = 1)
    θ = β * (layer.θ .+ inputs)
    γ = β * broadlike(layer.γ, inputs)
    Δ = β * broadlike(layer.Δ, inputs)
    ξ = broadlike(layer.ξ, inputs)
    return xReLU(promote(θ, γ, Δ, ξ)...)
end
