@doc raw"""
    wsum(A, wts)

Weighted sum of `A` along its trailing dimensions, weighted by `wts`.

```math
\sum_i A_i w_i
```

The reduced dimensions are inferred from the shape of `wts`, which must match
the trailing dimensions of `A` (all of `A` for a full reduction, returning a
scalar). Reduced dimensions are dropped from the result.
"""
function wsum(A::AbstractArray, wts::AbstractArray)
    kept = ndims(A) - ndims(wts)
    @assert kept ≥ 0
    @assert size(wts) == ntuple(i -> size(A, kept + i), ndims(wts))
    if kept == 0
        return _wsum_all(A, wts)
    else
        # A matmul-shaped reduction, which never materializes a weighted copy
        # of `A` and is a plain `sum` for lazy uniform `Ones` weights.
        S = _wsum_trailing(reshape(A, :, length(wts)), vec(wts))
        return reshape(S, ntuple(d -> size(A, d), Val(ndims(A) - ndims(wts))))
    end
end

# Float the operands (a no-op for float arrays): narrow-integer data and
# weights would otherwise accumulate in their own narrow type (wrapping on
# overflow), and mixed integer/float operands would take a different matmul
# kernel than float-converted copies, breaking exact reproducibility.
_asfloat(A::AbstractArray{<:AbstractFloat}) = A
_asfloat(A::AbstractArray) = float.(A)
# The default lazy uniform weights are exact ones already; keep them lazy (and
# `Bool`) so they cannot promote the eltype of the reduction.
_asfloat(A::Trues) = A

# `transpose`, not `dot`: the documented sum is `Σ Aᵢwᵢ`, without conjugation
_wsum_all(A::AbstractArray, wts::AbstractArray) = transpose(_asfloat(vec(A))) * _asfloat(vec(wts))
# uniform `Ones` weights reduce as a plain sum: the lazy CPU fill would take a
# scalar-indexing kernel in the mixed matmul when `A` is a GPU array
_wsum_all(A::AbstractArray, ::Ones{<:Real}) = sum(_asfloat(A))

_wsum_trailing(A::AbstractMatrix, wts::AbstractVector) = _asfloat(A) * _asfloat(wts)
# uniform `Ones` weights reduce as a plain sum (keeps the unweighted training
# path free of weighted copies and eltype promotion)
_wsum_trailing(A::AbstractMatrix, ::Ones{<:Real}) = sum(_asfloat(A); dims = 2)

# `A * Diagonal(vec(wts)) * B'`, the weighted outer product `Σᵢ wᵢ A[:,i] B[:,i]'`.
# Uniform `Ones` weights reduce to the plain product: `Diagonal` of a lazy CPU
# fill takes a scalar-indexing kernel when the factors are GPU arrays.
_weighted_outer(A::AbstractMatrix, wts::AbstractArray, B::AbstractMatrix) =
    A * Diagonal(_asfloat(vec(wts))) * B'
_weighted_outer(A::AbstractMatrix, ::Ones{<:Real}, B::AbstractMatrix) = A * B'

@doc raw"""
    wmean(A; wts)

Weighted mean of `A` along its trailing dimensions, weighted by `wts` (see
[`wsum`](@ref)). By default, lazy uniform weights over all of `A`, which
reduce like an ordinary `mean` without allocating a weights array or
promoting eltypes.

```math
\frac{\sum_i A_i w_i}{\sum_i w_i}
```
"""
function wmean(A::AbstractArray; wts::AbstractArray = Trues(size(A)))
    return wsum(A, wts) / sum(wts)
end

"""
    generate_sequences(n, A = 0:1)

Retruns an iterator over all sequences of length `n` out of the alphabet `A`.
"""
function generate_sequences(n::Int, A = 0:1)
    return (collect(seq) for seq in Iterators.product(ntuple(_ -> A, n)...))
end

# convert eltype before matrix multiply, to make sure we hit BLAS
convert_eltype(::Type{T}, A::AbstractArray) where {T} = convert(AbstractArray{T}, A)
with_eltype_of(X::AbstractArray, Y::AbstractArray) = convert_eltype(eltype(X), Y)

"""
    reshape_maybe(x, shape)

Like `reshape(x, shape)`, except that zero-dimensional outputs are returned as scalars.
"""
reshape_maybe(x::Number, ::Tuple{}) = x
reshape_maybe(x::AbstractArray, ::Tuple{}) = only(x)
reshape_maybe(x::AbstractArray, sz::Dims) = reshape(x, sz)
reshape_maybe(x::Union{Number, AbstractArray}, sz::Int...) = reshape(x, sz)

zeros_like(A::AbstractArray) = zeros_like(A, size(A))
zeros_like(A::AbstractArray, size) = zero(similar(A, size))

# mutable copy preserving the array backend (e.g. CuArray), materializing
# immutable/lazy arrays such as FillArrays.Zeros
_mutable_copy(A::AbstractArray) = copyto!(similar(A), A)
ones_like(A::AbstractArray, size) = one.(similar(A, size))
