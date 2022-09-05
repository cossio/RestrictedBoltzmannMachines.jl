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
    @assert size(θ) == size(γ)
    @assert size(w) == (size(g)..., size(θ)...)
    return RBM(Spin(; θ = g), Gaussian(; θ, γ), w)
end

function HopfieldRBM(g::AbstractArray{T,N}, w::AbstractArray) where {T,N}
    @assert size(g) == size(w)[1:N]
    m = size(w)[(N + 1):end]
    θ = zeros(eltype(g), m)
    γ = ones(eltype(g), m)
    return HopfieldRBM(g, θ, γ, w)
end

function HopfieldRBM(::Type{T}, n::Union{Int,Dims}, m::Union{Int,Dims}) where {T}
    g = zeros(T, n)
    θ = zeros(T, m)
    γ = ones(T, m)
    w = zeros(T, n..., m...)
    return HopfieldRBM(g, θ, γ, w)
end

HopfieldRBM(n::Union{Int,Dims}, m::Union{Int,Dims}) = HopfieldRBM(Float64, n, m)
