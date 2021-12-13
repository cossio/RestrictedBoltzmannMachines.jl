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
    weighted_mean(v, w)

Mean of `v` with weights `w`.
"""
function weighted_mean(v::AbstractArray, w::AbstractArray; dims = :)
    return mean(w .* v; dims = dims) / mean(w; dims = dims)
end

# faster special case
function weighted_mean(v::AbstractArray, w::Real = true; dims = :)
    return mean(v; dims = dims)
end

"""
    generate_sequences(n, A = 0:1)

Retruns an iterator over all sequences of length `n` out of the alphabet `A`.
"""
function generate_sequences(n::Int, A = 0:1)
    return (collect(seq) for seq in Iterators.product(ntuple(Returns(A), n)...))
end

"""
    tuplen(Val(N))

Constructs the tuple `(1, 2, ..., N)`.
"""
@generated tuplen(::Val{N}) where {N} = ntuple(identity, Val(N))
tuplen(N) = ntuple(identity, N)
