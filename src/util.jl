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

"""
    wmean(A; wts = nothing, dims = :)

Weighted mean of `A` along dimensions `dims`, with weights `wts`.
"""
function wmean(A::AbstractArray; wts::Union{AbstractArray,Nothing} = nothing, dims = :)
    if isnothing(wts)
        return mean(A; dims = dims)
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
