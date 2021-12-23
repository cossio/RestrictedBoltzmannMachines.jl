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

function energies(layer::xReLU, x::AbstractArray)
    energies(dReLU(layer), x)
end

cgfs(layer::xReLU) = cgfs(dReLU(layer))

function transfer_sample(layer::xReLU)
    return transfer_sample(dReLU(layer))
end

function effective(layer::xReLU, inputs, β::Real = 1)
    θ = β * (layer.θ .+ inputs)
    Δ = β * broadlike(layer.Δ, inputs)
    γ = β * broadlike(layer.γ, inputs)
    ξ = broadlike(layer.ξ, inputs)
    return xReLU(promote(θ, Δ, γ, ξ)...)
end
