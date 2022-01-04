struct pReLU{A<:AbstractArray}
    θ::A
    γ::A
    Δ::A
    η::A
    function pReLU(θ::A, γ::A, Δ::A, η::A) where {A<:AbstractArray}
        @assert size(θ) == size(γ) == size(Δ) == size(η)
        return new{A}(θ, γ, Δ, η)
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

function effective(layer::pReLU, inputs; β::Real = true)
    θ = β * (layer.θ .+ inputs)
    γ = β * broadlike(layer.γ, inputs)
    Δ = β * broadlike(layer.Δ, inputs)
    η = broadlike(layer.η, inputs)
    return pReLU(promote(θ, γ, Δ, η)...)
end

function ∂free_energy(layer::pReLU)
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

    ∂θ = @. -(pp * μp + pn * μn)
    ∂γ = @. (pp * μ2p / (1 + layer.η) + pn * μ2n / (1 - layer.η)) / 2
    ∂Δ = @. -(pp * μp / (1 + layer.η) - pn * μn / (1 - layer.η))
    ∂η = @. (
        pp * (-layer.γ * μ2p / 2 + layer.Δ * μp) / (1 + layer.η)^2 +
        pn * ( layer.γ * μ2n / 2 + layer.Δ * μn) / (1 - layer.η)^2
    )

    return (θ = ∂θ, γ = ∂γ, Δ = ∂Δ, η = ∂η)
end

function ∂energies(layer::pReLU, x::AbstractArray)
    @assert size(x) == size(layer) || size(x) == (size(layer)..., size(x)[end])

    xp = max.(x, 0)
    xn = min.(x, 0)

    ∂θ = @. -x
    ∂γ = @. (xp^2 / (1 + layer.η) + xn^2 / (1 - layer.η)) / 2
    ∂Δ = @. -(xp / (1 + layer.η) - xn / (1 - layer.η))
    ∂η = @. (
        (-layer.γ * xp^2 / 2 + layer.Δ * xp) / (1 + layer.η)^2 +
        ( layer.γ * xn^2 / 2 + layer.Δ * xn) / (1 - layer.η)^2
    )
    return (θ = ∂θ, γ = ∂γ, Δ = ∂Δ, η = ∂η)
end
