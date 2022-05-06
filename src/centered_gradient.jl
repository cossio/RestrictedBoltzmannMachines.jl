"""
    center_gradient(rbm, ∂, λv, λh)

Given a gradient `∂` of `rbm`, returns the gradient of the equivalent centered RBM
with `offset_v` and `offset_h`.
"""
function center_gradient(rbm::RBM, ∂::NamedTuple, offset_v::AbstractArray, offset_h::AbstractArray)
    @assert size(rbm.visible) == size(offset_v)
    @assert size(rbm.hidden) == size(offset_h)
    @assert size(∂.w) == size(rbm.w)
    Δv = grad2mean(rbm.visible, ∂.visible)
    Δh = grad2mean(rbm.hidden, ∂.hidden)
    ∂w = reshape(∂.w, length(rbm.visible), length(rbm.hidden))
    ∂wc = ∂w + vec(offset_v) * vec(Δh)' + vec(Δv) * vec(offset_h)'
    return (; ∂.visible, ∂.hidden, w = oftype(∂.w, reshape(∂wc, size(∂.w))))
end

"""
    uncenter_step(rbm, ∂, λv, λh)

Given parameter update step `∂` of a centered `rbm` with `offset_v` and `offset_h`, returns
the corresponding gradient of the equivalent uncentered RBM.
"""
function uncenter_step(rbm::RBM, ∂::NamedTuple, offset_v::AbstractArray, offset_h::AbstractArray)
    @assert size(rbm.visible) == size(offset_v)
    @assert size(rbm.hidden) == size(offset_h)
    @assert size(∂.w) == size(rbm.w)
    ∂w = reshape(∂.w, length(rbm.visible), length(rbm.hidden))
    shift_v = reshape(∂w  * vec(offset_h), size(rbm.visible))
    shift_h = reshape(∂w' * vec(offset_v), size(rbm.hidden))
    ∂v = uncenter_step(rbm.visible, ∂.visible, shift_v)
    ∂h = uncenter_step(rbm.hidden,  ∂.hidden,  shift_h)
    return (visible = oftype(∂.visible, ∂v), hidden = oftype(∂.hidden, ∂h), w = ∂.w)
end

function centered_gradient(
    rbm::RBM, ∂::NamedTuple, offset_v::AbstractArray, offset_h::AbstractArray
)
    ∂c = center_gradient(rbm, ∂, offset_v, offset_h)
    return uncenter_step(rbm, ∂c, offset_v, offset_h)
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
grad2mean(::Union{Binary,Spin,Potts,Gaussian,ReLU,pReLU,xReLU}, ∂::NamedTuple) = -∂.θ
grad2mean(::dReLU, ∂::NamedTuple) = -(∂.θp + ∂.θn)

grad2var(::Union{Binary,Potts}, ∂::NamedTuple) = -∂.θ .* (1 .+ ∂.θ)
grad2var(::Spin, ∂::NamedTuple) = (1 .- ∂.θ) .* (1 .+ ∂.θ)
grad2var(::Union{Gaussian,ReLU}, ∂::NamedTuple) = 2∂.γ - ∂.θ.^2
grad2var(::dReLU, ∂::NamedTuple) = 2 * (∂.γp + ∂.γn) - (∂.θp + ∂.θn).^2

function grad2var(l::pReLU, ∂::NamedTuple)
    @. 2l.η/l.γ * ((2l.Δ * ∂.Δ + l.η * ∂.η) * l.η - ∂.η - l.Δ * ∂.θ) + 2∂.γ * (1 + l.η^2) - ∂.θ^2
end

function grad2var(l::xReLU, ∂::NamedTuple)
    @. ((2∂.γ - ∂.θ^2) * l.γ - 2 * (∂.ξ + ∂.θ * l.Δ) * l.ξ + (4∂.γ * l.γ - ∂.θ^2 * l.γ + 4 * ∂.Δ * l.Δ) * l.ξ^2 - 4∂.ξ * l.ξ^3 + 2abs(l.ξ) * (2∂.γ * l.γ - ∂.θ^2 * l.γ - 3∂.ξ * l.ξ - ∂.θ * l.Δ * l.ξ)) / (l.γ * (1 + abs(l.ξ))^2)
end
