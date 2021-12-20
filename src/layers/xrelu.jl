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

energies(layer::xReLU, x::AbstractArray) = energies(dReLU(layer), x)
cgfs(layer::xReLU) = cgfs(dReLU(layer))

function sample(layer::xReLU)
    return sample(dReLU(layer))
end

function transform_layer(layer::xReLU, inputs, β::Real = 1)
    θ = β * (layer.θ .+ inputs)
    Δ = β * broadlike(layer.Δ, inputs)
    γ = β * broadlike(layer.γ, inputs)
    ξ = broadlike(layer.ξ, inputs)
    return xReLU(θ, Δ, γ, ξ)
end
