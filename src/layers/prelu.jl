struct pReLU{A<:AbstractArray}
    θ::A
    Δ::A
    γ::A
    η::A
    function pReLU(θ::A, Δ::A, γ::A, η::A) where {A<:AbstractArray}
        @assert size(θ) == size(γ) == size(Δ) == size(η)
        return new{A}(θ, Δ, γ, η)
    end
end

function pReLU(::Type{T}, n::Int...) where {T}
    θ = zeros(T, n...)
    Δ = zeros(T, n...)
    γ = ones(T, n...)
    η = zeros(T, n...)
    return pReLU(θ, Δ, γ, η)
end

pReLU(n::Int...) = pReLU(Float64, n...)

Flux.@functor pReLU

energy(layer::pReLU, x::AbstractArray) = energy(dReLU(layer), x)
cgf(layer::pReLU, inputs::AbstractArray, β::Real = 1) = cgf(dReLU(layer), inputs, β)

function sample_from_inputs(layer::pReLU, inputs::AbstractArray, β::Real = 1)
    return sample_from_inputs(dReLU(layer), inputs, β)
end
