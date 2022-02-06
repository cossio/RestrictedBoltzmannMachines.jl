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

@doc raw"""
    HopfieldRBM(g, θ, γ, w)
    HopfieldRBM(g, w)

Construct an RBM with spin visible units and Gaussian hidden units.
If not given, `θ = 0` and `γ = 1` by default.

```math
E(v, h) = -g'v - θ'h + \sum_\mu \frac{γ_\mu}{2} h_\mu^2 - v'wh
```
"""
function HopfieldRBM(g::AbstractArray, θ::AbstractArray, γ::AbstractArray, w::AbstractArray)
    @assert size(w) == (size(g)..., size(θ)...)
    @assert size(θ) == size(γ)
    return RBM(Spin(g), Gaussian(θ, γ), w)
end

function HopfieldRBM(g::AbstractArray, w::AbstractArray)
    @assert size(w)[1:ndims(g)] == size(g)
    return RBM(Spin(g), StdGauss(size(w)[(ndims(g) + 1):end]...), w)
end

function HopfieldRBM(::Type{T}, N::Union{Int,TupleN{Int}}, M::Union{Int,TupleN{Int}}) where {T}
    g = zeros(T, N...)
    θ = zeros(T, M...)
    γ = zeros(T, M...)
    w = zeros(T, N..., M...)
    return HopfieldRBM(g, θ, γ, w)
end

HopfieldRBM(N::Union{Int,TupleN{Int}}, M::Union{Int,TupleN{Int}}) = HopfieldRBM(Float64, N, M)
