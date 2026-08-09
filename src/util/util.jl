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
function wsum(A::AbstractArray{<:Any, N}, wts::AbstractArray{<:Real, N}) where {N}
    # full reduction: `wts` spans all of `A`
    @assert size(wts) == size(A)
    # `dot` conjugates its first argument, a no-op for the real `wts`, so the
    # documented sum `Σ Aᵢwᵢ` never conjugates `A`. `float` (identity on float
    # arrays) keeps integer/Bool samples on the same kernel as float-converted
    # runs, which reduce bit-identically.
    return dot(vec(wts), float(vec(A)))
end

# partial reduction over the trailing dimensions of `A`
function wsum(A::AbstractArray, wts::AbstractArray{<:Real})
    kept = ndims(A) - ndims(wts)
    @assert size(wts) == size(A)[(kept + 1):end]
    # A matmul-shaped reduction, which never materializes a weighted copy of `A`.
    # `float` (identity on float arrays) keeps integer/Bool samples on the same
    # matmul kernel as float-converted runs, which reduce bit-identically.
    S = float(reshape(A, :, length(wts))) * vec(wts)
    return reshape(S, ntuple(d -> size(A, d), Val(kept)))
end

# Uniform weights reduce as a plain sum: `float` per element accumulates
# non-float data in float without materializing a converted copy, and the
# lazy CPU fill would take a scalar-indexing kernel in the mixed matmul
# when `A` is a GPU array.
function wsum(A::AbstractArray{<:Any, N}, wts::Ones{<:Real, N}) where {N}
    @assert size(wts) == size(A)
    return sum(float, A)
end

function wsum(A::AbstractArray, wts::Ones{<:Real})
    kept = ndims(A) - ndims(wts)
    @assert size(wts) == size(A)[(kept + 1):end]
    S = sum(float, A; dims = (kept + 1):ndims(A))
    return reshape(S, ntuple(d -> size(A, d), Val(kept)))
end

# `A * Diagonal(vec(wts)) * B'`, the weighted outer product `Σᵢ wᵢ A[:,i] B[:,i]'`.
# Uniform `Ones` weights reduce to the plain product: `Diagonal` of a lazy CPU
# fill takes a scalar-indexing kernel when the factors are GPU arrays.
_weighted_outer(A::AbstractMatrix, wts::AbstractArray{<:Real}, B::AbstractMatrix) =
    A * Diagonal(vec(wts)) * B'
_weighted_outer(A::AbstractMatrix, ::Ones{<:Real}, B::AbstractMatrix) = A * B'

@doc raw"""
    wmean(A; [wts])

Weighted mean of `A` along its trailing dimensions, weighted by `wts` (see
[`wsum`](@ref)). By default, lazy uniform weights over all of `A`, which
reduce like an ordinary `mean` without allocating a weights array or
promoting eltypes.

```math
\frac{\sum_i A_i w_i}{\sum_i w_i}
```
"""
function wmean(A::AbstractArray; wts::AbstractArray{<:Real} = Trues(size(A)))
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
