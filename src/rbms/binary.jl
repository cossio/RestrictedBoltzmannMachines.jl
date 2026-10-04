@doc raw"""
    BinaryRBM(a, b, w)
    BinaryRBM(N, M)

Construct an RBM with binary visible and hidden units, which has an energy function:

```math
E(v, h) = -a'v - b'h - v'wh
```

Equivalent to `RBM(Binary(a), Binary(b), w)`.
"""
BinaryRBM(a::AbstractArray, b::AbstractArray, w::AbstractArray) = RBM(Binary(; θ = a), Binary(; θ = b), w)
BinaryRBM(::Type{T}, N::Union{Int, Dims}, M::Union{Int, Dims}) where {T} = BinaryRBM(zeros(T, N...), zeros(T, M...), zeros(T, N..., M...))
BinaryRBM(N::Union{Int, Dims}, M::Union{Int, Dims}) = BinaryRBM(Float64, N, M)
