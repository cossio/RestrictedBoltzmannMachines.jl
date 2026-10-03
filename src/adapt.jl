# Teach Adapt.jl to recurse through our structs, so that `adapt(CuArray, rbm)`,
# `cu(rbm)`, `adapt(Array, rbm)`, and other array backends work out of the box.
Adapt.@adapt_structure Binary
Adapt.@adapt_structure Spin
Adapt.@adapt_structure Potts
Adapt.@adapt_structure Gaussian
Adapt.@adapt_structure ReLU
Adapt.@adapt_structure dReLU
Adapt.@adapt_structure pReLU
Adapt.@adapt_structure xReLU
Adapt.@adapt_structure nsReLU
Adapt.@adapt_structure PottsGumbel
Adapt.@adapt_structure RBM
Adapt.@adapt_structure CenteredRBM
Adapt.@adapt_structure StandardizedRBM
Adapt.@adapt_structure ∂RBM

# Adapt.jl adaptor that copies every array, preserving its backend
struct _CopyArrays end
Adapt.adapt_storage(::_CopyArrays, x::AbstractArray) = copy(x)

# deep copy of a model (layers, weights, offsets and scales)
_copy_model(model) = Adapt.adapt(_CopyArrays(), model)

# copies the parameters of `src` into those of `dst`, a model of the same type
_copyto_model!(dst::AbstractArray, src::AbstractArray) = copyto!(dst, src)
function _copyto_model!(dst::T, src::T) where {T}
    foreach(f -> _copyto_model!(getfield(dst, f), getfield(src, f)), fieldnames(T))
    return dst
end
