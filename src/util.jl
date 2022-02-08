const TupleN{T,N} = NTuple{N,T}

# convenience functions to get generic Inf and NaN
inf(::Union{Type{T}, T}) where {T<:Number} = convert(T, Inf)
two(::Union{Type{T}, T}) where {T<:Number} = convert(T, 2)

"""
    wmean(A; wts = nothing, dims = :)

Weighted mean of `A` along dimensions `dims`, weighted by `wts`.
"""
function wmean(A::AbstractArray; wts::Union{AbstractArray,Nothing} = nothing, dims = :)
    if isnothing(wts)
        return Statistics.mean(A; dims = dims)
    elseif dims === (:)
        @assert size(wts) == size(A)
        return sum(A .* wts / sum(wts); dims = dims)
    else
        @assert size(wts) == size.(Ref(A), dims)
        wsz = ntuple(ndims(A)) do i
            i ∈ dims ? size(A, i) : 1
        end
        w = reshape(wts, wsz) / sum(wts)
        return sum(A .* w; dims = dims)
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
