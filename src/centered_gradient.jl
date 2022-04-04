"""
    center_gradient(rbm, ∂, λv, λh)

Given the gradient `∂` of `rbm`, returns the gradient of the equivalent centered RBM
with offsets `λv` and `λh`.
"""
function center_gradient(rbm::RBM, ∂::NamedTuple, λv::AbstractArray, λh::AbstractArray)
    @assert size(rbm.visible) == size(λv)
    @assert size(rbm.hidden) == size(λh)
    @assert size(∂.w) == size(rbm.w)
    Δv = grad2mean(rbm.visible, ∂.visible)
    Δh = grad2mean(rbm.hidden, ∂.hidden)
    ∂w = reshape(∂.w, length(rbm.visible), length(rbm.hidden))
    ∂wc = ∂w + vec(λv) * vec(Δh)' + vec(Δv) * vec(λh)'
    return (visible = ∂.visible, hidden = ∂.hidden, w = reshape(∂wc, size(∂.w)))
end

"""
    uncenter_step(rbm, ∂, λv, λh)

Given parameter update step `∂` of a centered `rbm` with offsets `λv, λh`, returns the
corresponding gradient of the equivalent uncentered RBM.
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
    return (visible = ∂v, hidden = ∂h, w = ∂.w)
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
    return (∂..., θ = ∂.θ - shift,)
end

function uncenter_step(layer::dReLU, ∂::NamedTuple, shift::AbstractArray)
    @assert size(layer) == size(∂.θp) == size(∂.θn) == size(shift)
    return (θp = ∂.θp - shift, θn = ∂.θn - shift, γp = ∂.γp, γn = ∂.γn)
end

# get moments from layer gradients, e.g. <v> = -derivative w.r.t. θ
grad2mean(::Union{Binary,Spin,Potts,Gaussian,ReLU,pReLU,xReLU}, ∂::NamedTuple) = -∂.θ
grad2mean(::dReLU, ∂::NamedTuple) = -(∂.θp + ∂.θn)

grad2var(::Union{Gaussian,ReLU}, ∂::NamedTuple) = 2∂.γ - ∂.θ.^2
grad2var(::dReLU, ∂::NamedTuple) = 2 * (∂.γp + ∂.γn) - (∂.θp + ∂.θn).^2
grad2var(l::Spin, ∂::NamedTuple) = 1 .- grad2mean(l, ∂).^2

function grad2var(l::Union{Binary,Potts}, ∂::NamedTuple)
    p = grad2mean(l, ∂)
    return p .* (1 .- p)
end
