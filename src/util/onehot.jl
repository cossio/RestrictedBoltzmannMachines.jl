"""
    onehot_encode(A, code)

Given an array `A` of `N` dimensions, returns a one-hot encoded `BitArray` of
`N + 1` dimensions where single entries of the first dimension are one.
"""
function onehot_encode(A::AbstractArray, code = sort(unique(A)))
    return reshape(A, 1, size(A)...) .== code
end

"""
    onehot_decode(X)

Given a onehot encoded array `X` of `N + 1` dimensions, returns the
equivalent categorical array of `N` dimensions.
"""
function onehot_decode(X::AbstractArray)
    idx = dropdims(argmax(X; dims=1); dims=1)
    return first.(Tuple.(idx))
end

"""
    categorical_sample_from_logits(logits)

Given a logits array `logits` of size `(q, *)` (where `q` is the number of classes),
returns an array `X` of size `(*)`, such that `X[i]` is a categorical random sample
from the distribution with logits `logits[:,i]`.
"""
function categorical_sample_from_logits(logits::AbstractArray)
    p = softmax(logits; dims=1)
    return categorical_sample(p)
end

"""
    categorical_sample_from_logits_gumbel(logits)

Like categorical_sample_from_logits, but using the Gumbel trick.
"""
function categorical_sample_from_logits_gumbel(logits::AbstractArray)
    z = logits .+ randgumbel.(eltype(logits))
    idx = dropdims(argmax(z; dims=1); dims=1)
    c = first.(Tuple.(idx))
    return c
end

"""
    categorical_sample(P)

Given a probability array `P` of size `(q, *)`, returns an array
`C` of size `(*)`, such that `C[i] ∈ 1:q` is a random sample from the
categorical distribution `P[:,i]`.
You must ensure that `P` defines a proper probability distribution.
"""
function categorical_sample(P::AbstractArray)
    idx = CartesianIndices(tail(size(P)))
    C = Array{Int}(undef, tail(size(P)))
    @inbounds for i in idx
        ps = @view P[:,i]
        C[i] = categorical_rand(ps)
    end
    return C
end

"""
    categorical_rand(ps)

Randomly draw `i` with probability `ps[i]`.
You must ensure that `ps` defines a proper probability distribution.
"""
function categorical_rand(ps::AbstractVector)
    i = 0
    u = rand(eltype(ps))
    for p in ps
        u -= p
        i += 1
        u ≤ 0 && break
    end
    return i
end

"""
    randgumbel(T = Float64)

Generates a random Gumbel variate.
"""
randgumbel(::Type{T} = Float64) where {T} = -log(randexp(T))
