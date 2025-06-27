"""
    ∂regularize!(∂, rbm; l2_fields = 0, l1_weights = 0, l2_weights = 0, l2l1_weights = 0)

Updates RBM gradients `∂`, with the regularization gradient.
"""
function ∂regularize!(
    ∂::∂RBM, # unregularized gradient
    rbm::RBM;
    l2_fields::Real = 0, # L2 regularization of visible unit fields
    l1_weights::Real = 0, # L1 regularization of weights
    l2_weights::Real = 0, # L2 regularization of weights
    l2l1_weights::Real = 0, # L2/L1 regularziation of weights (10.7554/eLife.39397, Eq. 8)
    zerosum::Bool = false # whether to zerosum gradients
)
    if !iszero(l2_fields)
        ∂regularize_fields!(∂.visible, rbm.visible; l2_fields)
    end
    if !iszero(l1_weights)
        ∂.w .+= l1_weights * sign.(rbm.w)
    end
    if !iszero(l2_weights)
        ∂.w .+= l2_weights * rbm.w
    end
    if !iszero(l2l1_weights)
        dims = ntuple(identity, ndims(rbm.visible))
        ∂.w .+= l2l1_weights * sign.(rbm.w) .* mean(abs, rbm.w; dims)
    end
    zerosum && zerosum!(∂, rbm)
    return ∂
end

function ∂regularize_fields!(
    ∂::AbstractArray, layer::Union{Binary,Spin,Potts,Gaussian,ReLU,xReLU,pReLU}; l2_fields::Real = 0
)
    if !iszero(l2_fields)
        ∂[1, ..] .+= l2_fields * layer.θ
    end
    return ∂
end

function ∂regularize_fields!(∂::AbstractArray, layer::dReLU; l2_fields::Real = 0)
    if !iszero(l2_fields)
        ∂[1, ..] .+= l2_fields * layer.θp
        ∂[2, ..] .+= l2_fields * layer.θn
    end
    return ∂
end

function ∂regularize(
    rbm::RBM;
    l2_fields::Real = 0, # L2 regularization of visible unit fields
    kw... # weights penalties
)
    visible = ∂regularize_fields(rbm.visible; l2_fields)
    w = ∂regularize_weights(rbm; kw...)
    return ∂RBM(visible, zero(rbm.hidden.par), w)
end

function ∂regularize_fields(layer::Union{Binary,Spin,Potts}; l2_fields::Real = 0)
    ∂θ = l2_fields * layer.θ
    return vstack((∂θ,))
end

function ∂regularize_fields(layer::Union{Gaussian,ReLU}; l2_fields::Real = 0)
    ∂θ = l2_fields * layer.θ
    ∂γ = zero(layer.γ)
    return vstack((∂θ, ∂γ))
end

function ∂regularize_fields(layer::dReLU; l2_fields::Real = 0)
    ∂θp = l2_fields * layer.θp
    ∂θn = l2_fields * layer.θn
    ∂γn = zero(layer.γn)
    ∂γp = zero(layer.γp)
    return vstack((∂θp, ∂θn, ∂γp, ∂γn))
end

function ∂regularize_fields(layer::pReLU; l2_fields::Real = 0)
    ∂θ = l2_fields * layer.θ
    ∂γ = zero(layer.γ)
    ∂Δ = zero(layer.Δ)
    ∂η = zero(layer.η)
    return vstack((∂θ, ∂γ, ∂Δ, ∂η))
end

function ∂regularize_fields(layer::xReLU; l2_fields::Real = 0)
    ∂θ = l2_fields * layer.θ
    ∂γ = zero(layer.γ)
    ∂Δ = zero(layer.Δ)
    ∂ξ = zero(layer.ξ)
    return vstack((∂θ, ∂γ, ∂Δ, ∂ξ))
end

function ∂regularize_weights(
    rbm::RBM;
    l1_weights::Real = 0, # L1 regularization of weights
    l2_weights::Real = 0, # L2 regularization of weights
    l2l1_weights::Real = 0 # L2/L1 regularziation of weights (10.7554/eLife.39397, Eq. 8)
)
    dims = ntuple(identity, ndims(rbm.visible))
    ∂l2l1 = l2l1_weights * sign.(rbm.w) .* mean(abs, rbm.w; dims)
    ∂l1 = l1_weights * sign.(rbm.w)
    ∂l2 = l2_weights * rbm.w
    return ∂l2l1 + ∂l1 + ∂l2
end

function regularization_penalty(rbm::RBM; l1_weights::Real = 0, l2_weights::Real = 0, l2l1_weights::Real = 0, l2_fields::Real = 0)
    dims = ntuple(identity, ndims(rbm.visible))
    N = length(rbm.visible)

    reg_fields = l2_fields/2 * sum(abs2, rbm.visible.θ)
    reg_l1_weights = l1_weights * sum(abs, rbm.w)
    reg_l2_weights = l2_weights/2 * sum(abs2, rbm.w)
    reg_l2l1_weights = l2l1_weights/(2N) * sum(abs2, sum(abs, rbm.w; dims))

    return reg_fields + reg_l1_weights + reg_l2_weights + reg_l2l1_weights
end
