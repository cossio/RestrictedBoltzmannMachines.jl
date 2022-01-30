"""
    L1L2(rbm)

L1/L2 squared norm of `rbm.w`.
Visible unit dimensions are reduced with L1 norm, while
hidden unit dimensions are reduced with L2 norm.
Note that no square root is taken.
"""
function L1L2(rbm::RBM)
    dims = ntuple(identity, ndims(rbm.visible))
    L1 = Statistics.mean(abs, rbm.w; dims=dims)
    L2 = Statistics.mean(abs2, L1)
    return L2
end

"""
    pgm_reg(rbm; λv, λw)

Regularization used on https://github.com/jertubiana/PGM.
"""
function pgm_reg(
    rbm::RBM{V}; λv::Real = 1//10^4, λw::Real = 1//10
) where {V<:Union{Binary,Spin,Potts}}
    fields_l2 = Statistics.mean(abs2, rbm.visible.θ)
    weights_l1l2 = L1L2(rbm)
    return λv/2 * fields_l2 + λw/2 * weights_l1l2
end
