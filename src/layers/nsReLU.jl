"""
    nsReLU(; θ, Δ, ξ)

A variant of `xReLU` units without scale parameter γ (which is fixed at 1). This is done
to remove the gauge invariance between the weights and the hidden units scale.

`nsReLU{N,A}` is an alias for `xReLU{N,A,true}`, where the last type parameter is the
`Bool` toggle that fixes γ = 1. Its `par` array stores only θ, Δ, ξ, while `layer.γ`
returns a lazy array of ones.
"""
const nsReLU{N,A} = xReLU{N,A,true}

nsReLU(par::AbstractArray) = xReLU(par; fix_γ = true)
nsReLU(; θ, Δ, ξ) = xReLU(; θ, Δ, ξ)
nsReLU(::Type{T}, sz::Dims) where {T} = xReLU(T, sz; fix_γ = true)
nsReLU(sz::Dims) = nsReLU(Float64, sz)

Base.propertynames(::nsReLU) = (:θ, :Δ, :ξ)

xReLU(layer::nsReLU) = xReLU(; layer.θ, γ = one.(layer.θ), layer.Δ, layer.ξ)
