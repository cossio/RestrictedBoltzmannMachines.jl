# allows me to write AbstractTensor{N} instead of AbstractArray{<:Any,N}
const AbstractTensor{N,T} = AbstractArray{T,N}

# convenience functions to get generic Inf and NaN
inf(::Union{Type{T}, T}) where {T} = convert(T, Inf)
two(::Union{Type{T}, T}) where {T} = convert(T, 2)

"""
    sum_(A; dims)

Sums `A` over dimensions `dims` and drops them.
"""
sum_(A::AbstractArray; dims) = dropdims(sum(A; dims=dims); dims=dims)

"""
    mean_(A; dims)

Takes the mean of `A` across dimensions `dims` and drops them.
"""
mean_(A::AbstractArray; dims) = dropdims(mean(A; dims=dims); dims=dims)

"""
    var_(A; dims)

Takes the variance of `A` across dimensions `dims` and drops them.
"""
function var_(A::AbstractArray; corrected::Bool=true, mean=nothing, dims)
    return dropdims(var(A; corrected=corrected, mean=mean, dims=dims); dims=dims)
end

"""
    std_(A; dims)

Takes the standard deviation of `A` across dimensions `dims` and drops them.
"""
function std_(A::AbstractArray; corrected::Bool=true, mean=nothing, dims)
    return dropdims(std(A; corrected=corrected, mean=mean, dims=dims); dims=dims)
end

const Wts = Union{AbstractVector, Nothing}

"""
    batch_mean(x, [wts])

Mean of `x` (over its last dimension), with (optional) weights `wts`,
dropping the reduced dimension.
"""
function batch_mean(x::AbstractTensor, wts::AbstractVector)
    if ndims(x) == 0
        return batch_mean_zerodims_error()
    elseif ndims(x) == 1
        return dot(x, wts / sum(wts))::Number
    else
        @assert ndims(x) ≥ 2
        @assert size(x) == (Base.front(size(x))..., length(wts))
        xmat = reshape(x, :, length(wts))
        xw = xmat * wts / sum(wts)
        result = reshape(xw, Base.front(size(x)))
        @assert ndims(result) == ndims(x) - 1 ≥ 1
        return result::AbstractArray
    end
end

function batch_mean(x::AbstractTensor, ::Nothing = nothing)
    if ndims(x) == 0
        return batch_mean_zerodims_error()
    elseif ndims(x) == 1
        return mean(x)::Number
    else
        @assert ndims(x) ≥ 2
        xm = dropdims(mean(x; dims=ndims(x)), dims=ndims(x))
        @assert ndims(xm) ≥ 1
        return xm
    end
end

function batch_mean_zerodims_error()
    throw(ArgumentError("Zero-dimensional arrays not supported"))
end

"""
    generate_sequences(n, A = 0:1)

Retruns an iterator over all sequences of length `n` out of the alphabet `A`.
"""
function generate_sequences(n::Int, A = 0:1)
    return (collect(seq) for seq in Iterators.product(ntuple(Returns(A), n)...))
end

"""
    broadlike(A, B...)

Broadcasts `A` into the size of `A .+ B .+ ...` (without actually doing a sum).
"""
broadlike(A, B...) = first_argument.(A, B...)
first_argument(x, y...) = x
