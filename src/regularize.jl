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
        ∂::AbstractArray, layer::Union{Binary, Spin, Potts, PottsGumbel, Gaussian, ReLU, xReLU, pReLU, nsReLU}; l2_fields::Real = 0
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

∂regularize_fields(layer::AbstractLayer; l2_fields::Real = 0) =
    ∂regularize_fields!(zero(layer.par), layer; l2_fields)

function regularization_penalty(rbm::RBM; l1_weights::Real = 0, l2_weights::Real = 0, l2l1_weights::Real = 0, l2_fields::Real = 0)
    dims = ntuple(identity, ndims(rbm.visible))
    N = length(rbm.visible)

    reg_fields = l2_fields / 2 * regularization_penalty_fields(rbm.visible)
    reg_l1_weights = l1_weights * sum(abs, rbm.w)
    reg_l2_weights = l2_weights / 2 * sum(abs2, rbm.w)
    reg_l2l1_weights = l2l1_weights / (2N) * sum(abs2, sum(abs, rbm.w; dims))

    return reg_fields + reg_l1_weights + reg_l2_weights + reg_l2l1_weights
end

regularization_penalty_fields(layer::dReLU) = sum(abs2, layer.θp) + sum(abs2, layer.θn)
regularization_penalty_fields(layer::Union{Binary, Spin, Potts, PottsGumbel, Gaussian, ReLU, pReLU, xReLU, nsReLU}) = sum(abs2, layer.θ)
