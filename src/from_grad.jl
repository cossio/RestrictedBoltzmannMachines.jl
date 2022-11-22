# get moments from layer cgf gradients, e.g. <v> = derivative of cgf w.r.t. θ

grad2ave(::Union{Binary,Spin,Potts,Gaussian,ReLU,pReLU,xReLU}, ∂::AbstractArray) = ∂[1, ..]
grad2ave(::dReLU, ∂::AbstractArray) = ∂[1, ..] + ∂[2, ..]

grad2var(::Union{Binary,Potts}, ∂::AbstractArray) = ∂[1, ..] .* (1 .- ∂[1, ..])
grad2var(::Spin, ∂::AbstractArray) = (1 .- ∂[1, ..]) .* (1 .+ ∂[1, ..])

function grad2var(l::Union{Gaussian,ReLU}, ∂::AbstractArray)
    ∂θ = @view ∂[1, ..]
    ∂γ = @view ∂[2, ..]
    return -2∂γ .* sign.(l.γ) - ∂θ.^2
end

function grad2var(l::dReLU, ∂::AbstractArray)
    ∂θp = ∂[1, ..]
    ∂θn = ∂[2, ..]
    ∂γp = ∂[3, ..]
    ∂γn = ∂[4, ..]
    return -2 * (∂γp .* sign.(l.γp) + ∂γn .* sign.(l.γn)) - (∂θp + ∂θn).^2
end

function grad2var(l::pReLU, ∂::AbstractArray)
    ∂θ = -∂[1, ..]
    ∂γ = -∂[2, ..]
    ∂Δ = -∂[3, ..]
    ∂η = -∂[4, ..]

    abs_γ = abs.(l.γ)
    ∂absγ = ∂γ .* sign.(l.γ)

    return @. 2l.η/abs_γ * ((2l.Δ * ∂Δ + l.η * ∂η) * l.η - ∂η - l.Δ * ∂θ) + 2∂absγ * (1 + l.η^2) - ∂θ^2
end

function grad2var(l::xReLU, ∂::AbstractArray)
    ∂θ = -∂[1, ..]
    ∂γ = -∂[2, ..]
    ∂Δ = -∂[3, ..]
    ∂ξ = -∂[4, ..]

    abs_γ = abs.(l.γ)
    ∂absγ = ∂γ .* sign.(l.γ)

    ν = @. 2∂absγ - ∂θ^2
    return @. (ν * abs_γ - 2 * (∂ξ + ∂θ * l.Δ) * l.ξ + ((ν + 2∂absγ) * abs_γ + 4 * ∂Δ * l.Δ) * l.ξ^2 - 4∂ξ * l.ξ^3 + 2abs(l.ξ) * (ν * abs_γ - 3∂ξ * l.ξ - ∂θ * l.Δ * l.ξ)) / (abs_γ * (1 + abs(l.ξ))^2)
end
