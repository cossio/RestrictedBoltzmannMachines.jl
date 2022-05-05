const TupleN{T,N} = NTuple{N,T}

# convenience functions to get generic Inf and NaN
inf(::Union{Type{T}, T}) where {T<:Number} = convert(T, Inf)
two(::Union{Type{T}, T}) where {T<:Number} = convert(T, 2)

@doc raw"""
    wmean(A; wts = nothing, dims = :)

Weighted mean of `A` along dimensions `dims`, weighted by `wts`.

```math
\frac{\sum_i A_i w_i}{\sum_i w_i}
```
"""
function wmean(A::AbstractArray; wts::Union{AbstractArray,Nothing} = nothing, dims = :)
    if isnothing(wts)
        # if no weights are given, fallback to unweighted mean
        return mean(A; dims)
    end

    if dims === (:)
        @assert size(wts) == size(A)
        w = wts
    else
        @assert size(wts) == ntuple(d -> size(A, dims[d]), length(dims))
        # insert singleton dimensions in weights, corresponding to reduced dimensions of `A`
        wsz = ntuple(ndims(A)) do i
            i ∈ dims ? size(A, i) : 1
        end
        w = reshape(wts, wsz)
    end

    return mean(A .* w; dims) ./ mean(wts)
end

@doc raw"""
    wsum(A; wts = nothing, dims = :)

Weighted sum of `A` along dimensions `dims`, weighted by `wts`.

```math
\frac{\sum_i A_i w_i}
```
"""
function wsum(A::AbstractArray; wts::Union{AbstractArray,Nothing} = nothing, dims = :)
    if isnothing(wts)
        return sum(A; dims)
    else
        return wmean(A; wts, dims) * sum(wts)
    end
end

"""
    generate_sequences(n, A = 0:1)

Retruns an iterator over all sequences of length `n` out of the alphabet `A`.
"""
function generate_sequences(n::Int, A = 0:1)
    return (collect(seq) for seq in Iterators.product(ntuple(_ -> A, n)...))
end

"""
    broadlike(A, B...)

Broadcasts `A` into the size of `A .+ B .+ ...` (without actually doing a sum).
"""
broadlike(A, B...) = first_argument.(A, B...)
first_argument(x, y...) = x

# convert to common eltype before matrix multiply, to make sure we hit BLAS
activations_convert_maybe(::AbstractArray{T}, x::AbstractArray{T}) where {T<:AbstractFloat} = x
activations_convert_maybe(::AbstractArray{T}, x::AbstractArray) where {T<:AbstractFloat} = map(T, x)
activations_convert_maybe(::AbstractArray, x::AbstractArray) = x

"""
    reshape_maybe(x, shape)

Like `reshape(x, shape)`, except that zero-dimensional outputs are returned as scalars.
"""
reshape_maybe(x::Number, ::Tuple{}) = x
reshape_maybe(x::AbstractArray, ::Tuple{}) = only(x)
reshape_maybe(x::AbstractArray, sz::TupleN{Int}) = reshape(x, sz)
reshape_maybe(x::Union{Number,AbstractArray}, sz::Int...) = reshape(x, sz)

"""
    repeat_size(sz, r...)

Returns `size(repeat(A, r...))`, provided `size(A) == sz`.
"""
repeat_size(sz::NTuple{N,Int}, r::Int...) where {N} = repeat_size(sz, r)
repeat_size(n::NTuple{N,Int}, r::NTuple{R,Int}) where {N,R} = ntuple(max(N, R)) do d
    if d ≤ N && d ≤ R
        n[d] * r[d]
    elseif R < d ≤ N
        n[d]
    elseif N < d ≤ R
        r[d]
    end
end

sizedims(A::AbstractArray, dims::Int...) = sizedims(A, dims)
sizedims(A::AbstractArray, dims::Tuple{Vararg{Int}}) = map(d -> size(A, d), dims)
sizedims(A::AbstractArray, ::Colon) = size(A)
