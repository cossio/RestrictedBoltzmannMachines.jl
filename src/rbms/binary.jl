@doc raw"""
    BinaryRBM(a, b, w)
    BinaryRBM(N, M)

Construct an RBM with binary visible and hidden units, which has an energy function:

```math
E(v, h) = -a'v - b'h - v'wh
```

Equivalent to `RBM(Binary(a), Binary(b), w)`.
"""
function BinaryRBM(a::AbstractArray, b::AbstractArray, w::AbstractArray)
    @assert size(w) == (size(a)..., size(b)...)
    return RBM(Binary(a), Binary(b), w)
end

function BinaryRBM(::Type{T}, N::Union{Int,TupleN{Int}}, M::Union{Int,TupleN{Int}}) where {T}
    a = zeros(T, N...)
    b = zeros(T, M...)
    w = zeros(T, N..., M...)
    return BinaryRBM(a, b, w)
end

BinaryRBM(N::Union{Int,TupleN{Int}}, M::Union{Int,Tuple{Vararg{Int}}}) = BinaryRBM(Float64, N, M)
