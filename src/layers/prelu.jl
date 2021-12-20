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

function energies(layer::pReLU, x::AbstractArray)
    drelu = dReLU(layer)
    return energies(drelu, x)
end

function cgfs(layer::pReLU)
    return cgfs(dReLU(layer))
end

function sample(layer::pReLU)
    return sample(dReLU(layer))
end

function transform_layer(layer::pReLU, inputs, β::Real = 1)
    θ = β * (layer.θ .+ inputs)
    Δ = β * broadlike(layer.Δ, inputs)
    γ = β * broadlike(layer.γ, inputs)
    η = broadlike(layer.η, inputs)
    return pReLU(θ, Δ, γ, η)
end
