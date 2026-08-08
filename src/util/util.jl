@doc raw"""
    wmean(A; wts = nothing, dims = :)

Weighted mean of `A` along dimensions `dims`, weighted by `wts`.

```math
\frac{\sum_i A_i w_i}{\sum_i w_i}
```

The weights are a vector along the last (sample) dimension of `A`, which must
be included in the reduced dimensions. Non-finite entries propagate to the
result, even when their weight is zero.
"""
function wmean(A::AbstractArray; wts::Union{AbstractVector, Nothing} = nothing, dims = :)
    if isnothing(wts)
        # if no weights are given, fallback to unweighted mean
        return mean(A; dims)
    end
    @assert length(wts) == size(A, ndims(A))
    @assert dims === (:) || ndims(A) ∈ dims
    # broadcast the weights along the last (sample) dimension of `A`
    w = reshape(wts, ntuple(d -> d < ndims(A) ? 1 : length(wts), ndims(A)))
    return mean(A .* w; dims) ./ mean(wts)
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
