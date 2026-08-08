@doc raw"""
    wsum(A, wts; dims = :)

Weighted sum of `A` along dimensions `dims`, weighted by `wts`.

```math
\sum_i A_i w_i
```

`wts` has the shape of the reduced dimensions of `A` (of all of `A` for
`dims = :`, returning a scalar). Reduced dimensions are kept as singletons,
like `sum(A; dims)`.
"""
function wsum(A::AbstractArray, wts::AbstractArray; dims = :)
    dims === (:) && return _wsum_all(A, wts)
    rdims = Tuple(dims)
    @assert size(wts) == ntuple(i -> size(A, rdims[i]), length(rdims))
    kept = ndims(A) - length(rdims)
    if rdims == ntuple(i -> kept + i, length(rdims))
        # Trailing reduced dimensions are a matmul-shaped reduction, which
        # never materializes a weighted copy of `A`.
        S = _wsum_trailing(reshape(A, :, length(wts)), vec(wts))
        return reshape(S, ntuple(d -> d ≤ kept ? size(A, d) : 1, ndims(A)))
    else
        # insert singleton dimensions in weights, corresponding to reduced dimensions of `A`
        wsz = ntuple(ndims(A)) do i
            i ∈ rdims ? size(A, i) : 1
        end
        return sum(A .* reshape(wts, wsz); dims)
    end
end

_wsum_all(A::AbstractArray, wts::AbstractArray) = dot(vec(A), vec(wts))
_wsum_all(A::AbstractArray, ::Ones) = sum(A)
_wsum_trailing(A::AbstractMatrix, wts::AbstractVector) = A * wts
_wsum_trailing(A::AbstractMatrix, ::Ones) = sum(A; dims = 2)

@doc raw"""
    wmean(A; wts = Ones{Bool}(...), dims = :)

Weighted mean of `A` along dimensions `dims`, weighted by `wts`.

```math
\frac{\sum_i A_i w_i}{\sum_i w_i}
```

`wts` defaults to lazy uniform weights (`FillArrays.Ones`), which reduce like
an ordinary `mean` without allocating a weights array or promoting eltypes.
"""
function wmean(A::AbstractArray; dims = :, wts::AbstractArray = _uniform_wts(A, dims))
    return wsum(A, wts; dims) / sum(wts)
end

# Lazy uniform weights matching the reduced dimensions `dims` of `A`.
_uniform_wts(A::AbstractArray, ::Colon) = Ones{Bool}(size(A))
_uniform_wts(A::AbstractArray, dims) = Ones{Bool}(ntuple(i -> size(A, dims[i]), length(dims)))

# Scale the batch columns of `A` by their weights, for matmul-shaped weighted
# reductions. Fold the weights into the smaller factor of a product to keep the
# temporary small; lazy uniform `Ones` weights are a no-op.
_scale_obs(A::AbstractMatrix, wts::AbstractVector) = A .* reshape(wts, 1, :)
_scale_obs(A::AbstractMatrix, ::Ones) = A

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
