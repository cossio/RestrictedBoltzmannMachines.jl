"""
    ∂reg!(∂, rbm; l2_fields = 0, l1_weights = 0, l2_weights = 0, l2l1_weights = 0)

Updates `∂` with the regularization gradient.
Based on https://github.com/jertubiana/PGM.
"""
function ∂reg!(
    ∂::NamedTuple, rbm::RBM{<:Union{Binary,Spin,Gaussian,ReLU,xReLU,pReLU}};
    l2_fields::Real = 0, l1_weights::Real = 0, l2_weights::Real = 0, l2l1_weights::Real = 0
)
    if !iszero(l2_fields)
        ∂reg_fields!(∂.visible, visible(rbm); l2_fields)
    end
    if !iszero(l1_weights)
        ∂.w .+= l1_weights * sign.(weights(rbm))
    end
    if !iszero(l2_weights)
        ∂.w .+= l2_weights * weights(rbm)
    end
    if !iszero(l2l1_weights)
        vdims = ntuple(identity, ndims(visible(rbm)))
        ∂.w .+= l2l1_weights * sign.(weights(rbm)) .* mean(abs, weights(rbm); dims=vdims)
    end
    return ∂
end

function ∂reg_fields!(
    ∂::NamedTuple, layer::Union{Binary,Spin,Gaussian,ReLU,xReLU,pReLU}; l2_fields::Real = 0
)
    if !iszero(l2_fields)
        ∂.θ .+= l2_fields * layer.θ
    end
    return ∂
end

function ∂reg_fields!(∂::NamedTuple, layer::dReLU; l2_fields::Real = 0)
    if !iszero(l2_fields)
        ∂.θp .+= l2_fields * layer.θp
        ∂.θn .+= l2_fields * layer.θn
    end
    return ∂
end
