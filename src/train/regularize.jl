export jerome_regularization

"""
    jerome_regularization(rbm; λv, λw)

Based on 10.7554/eLife.39397.
"""
jerome_regularization(rbm::RBM; λv::Real = 1//10^4, λw::Real = 1//10) =
    λv/2 * fields_l2(rbm.vis) + λw/2 * weights_l1l2(rbm)

fields_l2(layer::Union{Potts,Binary,Spin,Gaussian,ReLU}) = l2(layer.θ)
weights_l1l2(rbm::RBM) = l1l2(rbm.weights, Val(ndims(rbm.vis)))
l2(A::AbstractArray{<:Real}) = mean(A.^2)
l1(A::AbstractArray{<:Real}) = mean(abs.(A))

"""
    l1l2(w::AbstractArray, Val(dim))

L1/L2 norm of `w`, which can be a tensor. Then dimensions ≤ `dim` will be
reduced with L1 norm and dimensions > `dim` with L2 norm (without taking
square root).
"""
function l1l2(w::AbstractArray{<:Any,N}, ::Val{dim}) where {N,dim}
    dims = OneHot.tuplen(Val(dim))
    return mean(mean(abs.(w); dims = dims).^2)
end

"""
    hidden_l1(rbm, datum)

Average L1 norm of hidden layer activity conditioned on configurations `datum.v`
of the visible layer. The average is taken by weighting configurations with
`datum.w`.
"""
function hidden_l1(rbm::RBM, datum::NamedTuple)
    Ih = inputs_v_to_h(rbm, datum.v)
    return hidden_l1(rbm, Ih, datum.w)
end
function hidden_l1(rbm::RBM, Ih::AbstractArray, w::AbstractArray)
    absh = hidden_l1(rbm, Ih)
    bdim = batchndims(rbm.hid, absh)
    return mean(tensormul_ll(absh, w, Val(bdim)))
end
hidden_l1(rbm::RBM, Ih::AbstractArray, w::Number = 1) = transfer_mean_abs(rbm.hid, Ih)

hidden_l1l2(rbm::RBM, datum::NamedTuple) = hidden_l1(rbm, datum)^2 / 2
hidden_l1l2(rbm::RBM, Ih::AbstractArray, w::Num) = hidden_l1(rbm, Ih, w)^2 / 2

function default_regularization(rbm::RBM, datum::NamedTuple;
                                λw::Real, λh::Real, λg::Real)
    hl1 = hidden_l1(rbm, datum)
    wl1l2 = weights_l1l2(rbm)
    gl2 = fields_l2(rbm.vis)
    return λh * hl1 + λg/2 * gl2 + λw/2 * wl1l2
end

#= gradient =#
