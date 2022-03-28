function center_gradient(rbm::RBM, ∂::NamedTuple, λv::AbstractArray, λh::AbstractArray)
    @assert size(rbm.visible) == size(λv)
    @assert size(rbm.hidden) == size(λh)
    @assert size(∂.w) == size(rbm.w)

    ∂wmat = reshape(∂.w, length(rbm.visible), length(rbm.hidden))
    ∂cwmat = ∂wmat - vec(λv) * vec(∂.hidden.θ)' - vec(∂.visible.θ) * vec(λh)'
    ∂cw = reshape(∂cwmat, size(rbm.w))

    shift_v = reshape(∂cwmat  * vec(λh), size(rbm.visible))
    shift_h = reshape(∂cwmat' * vec(λv), size(rbm.hidden))
    ∂cv = center_gradient(rbm.visible, ∂.visible, shift_v)
    ∂ch = center_gradient(rbm.hidden,  ∂.hidden,  shift_h)

    return (visible = ∂cv, hidden = ∂ch, w = ∂cw)
end

function center_gradient(
    layer::Union{Binary,Spin,Potts,Gaussian,ReLU,pReLU,xReLU},
    ∂::NamedTuple, λ::AbstractArray
)
    @assert size(layer) == size(∂.θ) == size(λ)
    return (∂..., θ = ∂.θ - λ,)
end

function center_gradient(layer::dReLU, ∂::NamedTuple, λ::AbstractArray)
    @assert size(layer) == size(∂.θp) == size(∂.θn) == size(λ)
    return (θp = ∂.θp - λ, θn = ∂.θn - λ, γp = ∂.γp, γn = ∂.γn)
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
