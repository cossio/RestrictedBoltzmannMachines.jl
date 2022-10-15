"""
    center_gradient(rbm, ∂, λv, λh)

Given the gradient `∂` of `rbm`, returns the gradient of the equivalent centered RBM
with offsets `λv` and `λh`.
"""
function center_gradient(rbm::RBM, ∂::∂RBM, λv::AbstractArray, λh::AbstractArray)
    ∂ = oftype(∂, center_gradient_v(rbm, ∂, λv))
    ∂ = oftype(∂, center_gradient_h(rbm, ∂, λh))
    return ∂
end

function center_gradient_v(rbm::RBM, ∂::∂RBM, λv::AbstractArray)
    @assert size(rbm.visible) == size(λv)
    @assert size(∂.w) == size(rbm.w)
    Δh = grad2ave(rbm.hidden, -∂.hidden)
    ∂w = reshape(∂.w, length(rbm.visible), length(rbm.hidden)) + vec(λv) * vec(Δh)'
    return ∂RBM(∂.visible, ∂.hidden, oftype(∂.w, reshape(∂w, size(∂.w))))
end

function center_gradient_h(rbm::RBM, ∂::∂RBM, λh::AbstractArray)
    @assert size(rbm.hidden) == size(λh)
    @assert size(∂.w) == size(rbm.w)
    Δv = grad2ave(rbm.visible, -∂.visible)
    ∂w = reshape(∂.w, length(rbm.visible), length(rbm.hidden)) + vec(Δv) * vec(λh)'
    return ∂RBM(∂.visible, ∂.hidden, oftype(∂.w, reshape(∂w, size(∂.w))))
end

"""
    uncenter_step(rbm, ∂, λv, λh)

Given parameter update step `∂` of a centered `rbm` with offsets `λv, λh`, returns
the corresponding gradient of the equivalent uncentered RBM.
"""
function uncenter_step(rbm::RBM, ∂::∂RBM, λv::AbstractArray, λh::AbstractArray)
    @assert size(rbm.visible) == size(λv)
    @assert size(rbm.hidden) == size(λh)
    @assert size(∂.w) == size(rbm.w)
    ∂w = reshape(∂.w, length(rbm.visible), length(rbm.hidden))
    shift_v = reshape(∂w  * vec(λh), size(rbm.visible))
    shift_h = reshape(∂w' * vec(λv), size(rbm.hidden))
    ∂v = uncenter_step(rbm.visible, ∂.visible, shift_v)
    ∂h = uncenter_step(rbm.hidden,  ∂.hidden,  shift_h)
    return ∂RBM(∂v, ∂h, ∂.w)
end

function centered_gradient(rbm::RBM, ∂::∂RBM, λv::AbstractArray, λh::AbstractArray)
    ∂c = center_gradient(rbm, ∂, λv, λh)
    return uncenter_step(rbm, ∂c, λv, λh)
end

function uncenter_step(layer::Union{Binary,Spin,Potts}, ∂::AbstractArray, shift::AbstractArray)
    @assert size(layer.par) == size(∂)
    ∂θ = ∂[1, ..]
    return oftype(∂, vstack((∂θ - shift,)))
end

function uncenter_step(layer::Union{Gaussian,ReLU}, ∂::AbstractArray, shift::AbstractArray)
    @assert size(layer.par) == size(∂)
    ∂θ = ∂[1, ..]
    ∂γ = ∂[2, ..]
    return oftype(∂, vstack((∂θ - shift, ∂γ)))
end

function uncenter_step(layer::pReLU, ∂::AbstractArray, shift::AbstractArray)
    @assert size(layer.par) == size(∂)
    ∂θ = ∂[1, ..]
    ∂γ = ∂[2, ..]
    ∂Δ = ∂[3, ..]
    ∂η = ∂[4, ..]
    return oftype(∂, vstack((∂θ - shift, ∂γ, ∂Δ, ∂η)))
end

function uncenter_step(layer::xReLU, ∂::AbstractArray, shift::AbstractArray)
    @assert size(layer.par) == size(∂)
    ∂θ = ∂[1, ..]
    ∂γ = ∂[2, ..]
    ∂Δ = ∂[3, ..]
    ∂ξ = ∂[4, ..]
    return oftype(∂, vstack((∂θ - shift, ∂γ, ∂Δ, ∂ξ)))
end

function uncenter_step(layer::dReLU, ∂::AbstractArray, shift::AbstractArray)
    @assert size(layer.par) == size(∂)
    ∂θp = ∂[1, ..]
    ∂θn = ∂[2, ..]
    ∂γp = ∂[3, ..]
    ∂γn = ∂[4, ..]
    return oftype(∂, vstack((∂θp - shift, ∂θn - shift, ∂γp, ∂γn)))
end

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
