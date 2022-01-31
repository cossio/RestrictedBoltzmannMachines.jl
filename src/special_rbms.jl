@doc raw"""
    BinaryRBM(a, b, w)
    BinaryRBM(N, M)

Construct an RBM with binary visible and hidden units, which has an energy function:

```math
E(v, h) = -a'v - b'h - v'wh
```

Equivalent to `RBM(Binary(a), Binary(b), w)`.
"""
function BinaryRBM end

function BinaryRBM(a::AbstractArray, b::AbstractArray, w::AbstractArray)
    @assert size(w) == (size(a)..., size(b)...)
    return RBM(Binary(a), Binary(b), w)
end

function BinaryRBM(N::Tuple{Vararg{Int}}, M::Tuple{Vararg{Int}}, ::Type{T} = Float64) where {T}
    a = zeros(T, N...)
    b = zeros(T, M...)
    w = zeros(T, N..., M...)
    return BinaryRBM(a, b, w)
end

BinaryRBM(N::Int, M::Int, ::Type{T} = Float64) where {T} = BinaryRBM((N,), (M,), T)

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

function HopfieldRBM(N::Tuple{Vararg{Int}}, M::Tuple{Vararg{Int}}, ::Type{T} = Float64) where {T}
    g = zeros(N...)
    θ = zeros(M...)
    γ = zeros(M...)
    w = zeros(N..., M...)
    return HopfieldRBM(g, θ, γ, w)
end

HopfieldRBM(N::Int, M::Int, ::Type{T} = Float64) where {T} = HopfieldRBM((N,), (M,), T)
