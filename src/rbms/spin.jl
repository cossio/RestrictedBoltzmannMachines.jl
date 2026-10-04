"""
    SpinRBM(a, b, w)

Construct an RBM with spin visible and hidden units.
Equivalent to `RBM(Spin(a), Spin(b), w)`.
"""
SpinRBM(a::AbstractArray, b::AbstractArray, w::AbstractArray) = RBM(Spin(; θ = a), Spin(; θ = b), w)
SpinRBM(::Type{T}, N::Union{Int, Dims}, M::Union{Int, Dims}) where {T} = SpinRBM(zeros(T, N...), zeros(T, M...), zeros(T, N..., M...))
SpinRBM(N::Union{Int, Dims}, M::Union{Int, Dims}) = SpinRBM(Float64, N, M)
