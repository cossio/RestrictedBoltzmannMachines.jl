function SpinRBM(a::AbstractArray, b::AbstractArray, w::AbstractArray)
    @assert size(w) == (size(a)..., size(b)...)
    visible = Spin(; θ = a)
    hidden = Spin(; θ = b)
    return RBM(visible, hidden, w)
end

function SpinRBM(::Type{T}, N::Union{Int,Dims}, M::Union{Int,Dims}) where {T}
    a = zeros(T, N...)
    b = zeros(T, M...)
    w = zeros(T, N..., M...)
    return SpinRBM(a, b, w)
end

SpinRBM(N::Union{Int,Dims}, M::Union{Int,Dims}) = SpinRBM(Float64, N, M)
