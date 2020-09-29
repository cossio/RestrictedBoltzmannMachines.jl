export no_regularization, jerome_regularization

no_regularization(obj) = 0

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
hidden_l1(rbm::RBM, datum::NamedTuple) = hidden_l1(rbm, datum.v, datum.w)

"""
    hidden_l1(rbm, v, w = 1)

Average L1 norm of hidden layer activity conditioned on configurations `v` of
the visible layer. The average is taken by weighting configurations with `w`.
"""
function hidden_l1(rbm::RBM, v::AbstractArray, w::Number = 1)
    Ivh = inputs_v_to_h(rbm, v)
    absh = transfer_mean_abs(rbm.hid, Ivh)
    return mean(absh)
end
function hidden_l1(rbm::RBM, v::AbstractArray, w::AbstractArray)
    Ivh = inputs_v_to_h(rbm, v)
    absh = transfer_mean_abs(rbm.hid, Ivh)
    bdim = batchndims(rbm.hid, absh)
    return mean(tensormul_ll(absh, w, Val(bdim)))
end

#= gradient =#
