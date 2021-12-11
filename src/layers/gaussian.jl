"""
    Gaussian(θ, γ)

Gaussian layer, with location parameters `θ` and scale parameters `γ`.
"""
struct Gaussian{A<:AbstractArray}
    θ::A
    γ::A
    function Gaussian(θ::A, γ::A) where {A<:AbstractArray}
        @assert size(θ) == size(γ)
        return new{A}(θ, γ)
    end
end

function Gaussian(::Type{T}, n::Int...) where {T}
    return Gaussian(zeros(T, n...), ones(T, n...))
end

Gaussian(n::Int...) = Gaussian(Float64, n...)

Flux.@functor Gaussian

function energy(layer::Gaussian, x::AbstractArray)
    @assert size(x) == (size(layer)..., size(x)[end])
    E = @. (abs(layer.γ) * x / 2 - layer.θ) * x
    return sum_(E; dims = layerdims(layer))
end

function sample_from_inputs(layer::Gaussian, inputs::AbstractArray)
    @assert size(inputs) == (size(layer)..., size(inputs)[end])
    μ = @. (layer.θ + inputs) / abs.(layer.γ)
    σ = @. inv(sqrt(abs(layer.γ)))
    z = randn(eltype(μ), size(μ))
    return μ .+ σ .* z
end

function sample_from_inputs(layer::Gaussian, inputs::AbstractArray, β::Real)
    layer_ = Gaussian(layer.θ .* β, layer.γ .* β)
    return sample_from_inputs(layer_, inputs .* β)
end

function cgf(layer::Gaussian, inputs::AbstractArray)
    @assert size(inputs) == (size(layer)..., size(inputs)[end])
    Γ = gauss_cgf.(layer.θ .+ inputs, layer.γ)
    return sum_(Γ; dims = layerdims(layer))
end

function cgf(layer::Gaussian, inputs::AbstractArray, β::Real)
    layer_ = Gaussian(layer.θ .* β, layer.γ .* β)
    return cgf(layer_, inputs .* β) / β
end

function gauss_cgf(θ::Real, γ::Real)
    γa = abs(γ)
    return θ^2 / 2γa - log(γa/π/2) / 2
end
