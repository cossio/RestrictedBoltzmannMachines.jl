"""
    center_gradient(rbm, ∂, λv, λh)

Given the gradient `∂` of `rbm`, returns the gradient of the equivalent centered RBM
with offsets `λv` and `λh`.
"""
function center_gradient(rbm::RBM, ∂::NamedTuple, λv::AbstractArray, λh::AbstractArray)
    ∂ = oftype(∂, center_gradient_v(rbm, ∂, λv))
    ∂ = oftype(∂, center_gradient_h(rbm, ∂, λh))
    return ∂
end

function center_gradient_v(rbm::RBM, ∂::NamedTuple, λv::AbstractArray)
    @assert size(rbm.visible) == size(λv)
    @assert size(∂.w) == size(rbm.w)
    Δh = grad2ave(rbm.hidden, ∂.hidden)
    ∂w = reshape(∂.w, length(rbm.visible), length(rbm.hidden)) + vec(λv) * vec(Δh)'
    return oftype(∂, (; ∂..., w = reshape(∂w, size(∂.w))))
end

function center_gradient_h(rbm::RBM, ∂::NamedTuple, λh::AbstractArray)
    @assert size(rbm.hidden) == size(λh)
    @assert size(∂.w) == size(rbm.w)
    Δv = grad2ave(rbm.visible, ∂.visible)
    ∂w = reshape(∂.w, length(rbm.visible), length(rbm.hidden)) + vec(Δv) * vec(λh)'
    return oftype(∂, (; ∂..., w = reshape(∂w, size(∂.w))))
end

"""
    uncenter_step(rbm, ∂, λv, λh)

Given parameter update step `∂` of a centered `rbm` with offsets `λv, λh`, returns
the corresponding gradient of the equivalent uncentered RBM.
"""
function uncenter_step(rbm::RBM, ∂::NamedTuple, λv::AbstractArray, λh::AbstractArray)
    @assert size(rbm.visible) == size(λv)
    @assert size(rbm.hidden) == size(λh)
    @assert size(∂.w) == size(rbm.w)
    ∂w = reshape(∂.w, length(rbm.visible), length(rbm.hidden))
    shift_v = reshape(∂w  * vec(λh), size(rbm.visible))
    shift_h = reshape(∂w' * vec(λv), size(rbm.hidden))
    ∂v = uncenter_step(rbm.visible, ∂.visible, shift_v)
    ∂h = uncenter_step(rbm.hidden,  ∂.hidden,  shift_h)
    return (visible = oftype(∂.visible, ∂v), hidden = oftype(∂.hidden, ∂h), w = ∂.w)
end

function centered_gradient(rbm::RBM, ∂::NamedTuple, λv::AbstractArray, λh::AbstractArray)
    ∂c = center_gradient(rbm, ∂, λv, λh)
    return uncenter_step(rbm, ∂c, λv, λh)
end

function uncenter_step(
    layer::Union{Binary,Spin,Potts,Gaussian,ReLU,pReLU,xReLU},
    ∂::NamedTuple, shift::AbstractArray
)
    @assert size(layer) == size(∂.θ) == size(shift)
    return (; ∂..., θ = ∂.θ - shift,)
end

function uncenter_step(layer::dReLU, ∂::NamedTuple, shift::AbstractArray)
    @assert size(layer) == size(∂.θp) == size(∂.θn) == size(shift)
    return (θp = ∂.θp - shift, θn = ∂.θn - shift, γp = ∂.γp, γn = ∂.γn)
end

# get moments from layer gradients, e.g. <v> = -derivative w.r.t. θ
grad2ave(::Union{Binary,Spin,Potts,Gaussian,ReLU,pReLU,xReLU}, ∂::NamedTuple) = -∂.θ
grad2ave(::dReLU, ∂::NamedTuple) = -(∂.θp + ∂.θn)

grad2var(::Union{Binary,Potts}, ∂::NamedTuple) = -∂.θ .* (1 .+ ∂.θ)
grad2var(::Spin, ∂::NamedTuple) = (1 .- ∂.θ) .* (1 .+ ∂.θ)
grad2var(l::Union{Gaussian,ReLU}, ∂::NamedTuple) = 2∂.γ .* sign.(l.γ) - ∂.θ.^2
grad2var(l::dReLU, ∂::NamedTuple) = 2 * (∂.γp .* sign.(l.γp) + ∂.γn .* sign.(l.γn)) - (∂.θp + ∂.θn).^2

function grad2var(l::pReLU, ∂::NamedTuple)
    abs_γ = abs.(l.γ)
    ∂absγ = ∂.γ .* sign.(l.γ)
    @. 2l.η/abs_γ * ((2l.Δ * ∂.Δ + l.η * ∂.η) * l.η - ∂.η - l.Δ * ∂.θ) + 2∂absγ * (1 + l.η^2) - ∂.θ^2
end

function grad2var(l::xReLU, ∂::NamedTuple)
    abs_γ = abs.(l.γ)
    ∂absγ = ∂.γ .* sign.(l.γ)
    ν = @. 2∂absγ - ∂.θ^2
    @. (ν * abs_γ - 2 * (∂.ξ + ∂.θ * l.Δ) * l.ξ + ((ν + 2∂absγ) * abs_γ + 4 * ∂.Δ * l.Δ) * l.ξ^2 - 4∂.ξ * l.ξ^3 + 2abs(l.ξ) * (ν * abs_γ - 3∂.ξ * l.ξ - ∂.θ * l.Δ * l.ξ)) / (abs_γ * (1 + abs(l.ξ))^2)
end
