struct xReLU{A<:AbstractArray}
    θ::A
    Δ::A
    γ::A
    ξ::A
    function xReLU(θ::A, Δ::A, γ::A, ξ::A) where {A<:AbstractArray}
        @assert size(θ) == size(γ) == size(Δ) == size(ξ)
        return new{A}(θ, Δ, γ, ξ)
    end
end

function xReLU(::Type{T}, n::Int...) where {T}
    θ = zeros(T, n...)
    Δ = zeros(T, n...)
    γ = ones(T, n...)
    ξ = zeros(T, n...)
    return pReLU(θ, Δ, γ, ξ)
end

xReLU(n::Int...) = xReLU(Float64, n...)

Flux.@functor xReLU

energy(layer::xReLU, x::AbstractArray) = energy(dReLU(layer), x)
cgf(layer::xReLU, inputs::AbstractArray, β::Real = 1) = cgf(dReLU(layer), inputs, β)

function sample_from_inputs(layer::xReLU, inputs::AbstractArray, β::Real = 1)
    return sample_from_inputs(dReLU(layer), inputs, β)
end
