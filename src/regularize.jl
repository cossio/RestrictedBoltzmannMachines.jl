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
    _∂regularize_weights!(∂.w, rbm; l1_weights, l2_weights, l2l1_weights)
    zerosum && zerosum!(∂, rbm)
    return ∂
end

function _∂regularize_weights!(
        ∂w::AbstractArray, rbm::RBM;
        l1_weights::Real, l2_weights::Real, l2l1_weights::Real,
        scale::AbstractArray = Ones{eltype(rbm.w)}(size(rbm.w))
    )
    if !iszero(l1_weights)
        ∂w .+= _maybe_div(l1_weights .* sign.(rbm.w), scale)
    end
    if !iszero(l2_weights)
        ∂w .+= _maybe_div(l2_weights .* rbm.w, scale)
    end
    if !iszero(l2l1_weights)
        dims = ntuple(identity, ndims(rbm.visible))
        ∂w .+= _maybe_div(l2l1_weights .* sign.(rbm.w) .* mean(abs, rbm.w; dims), scale)
    end
    return ∂w
end

function ∂regularize_fields!(∂::AbstractArray, layer::_ThetaLayers; l2_fields::Real = 0)
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

# zeros_like (not zero) so immutable layer parameter arrays get a mutable buffer
∂regularize_fields(layer::AbstractLayer; l2_fields::Real = 0) =
    ∂regularize_fields!(zeros_like(layer.par), layer; l2_fields)

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
regularization_penalty_fields(layer::_ThetaLayers) = sum(abs2, layer.θ)
