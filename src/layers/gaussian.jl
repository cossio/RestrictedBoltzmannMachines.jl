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
    @assert size(inputs) == (size(layer)..., size(inputs)[end])
    layer_ = Gaussian(layer.θ .* β, layer.γ .* β)
    return sample_from_inputs(layer_, inputs .* β)
end

function cgf(layer::Gaussian, inputs::AbstractArray)
    @assert size(inputs) == (size(layer)..., size(inputs)[end])
    Γ = @. (layer.θ + inputs)^2 / abs(2layer.γ) - log(abs(layer.γ)/π/2)/2
    return sum_(Γ; dims = layerdims(layer))
end

function cgf(layer::Gaussian, inputs::AbstractArray, β::Real)
    @assert size(inputs) == (size(layer)..., size(inputs)[end])
    layer_ = Gaussian(layer.θ .* β, layer.γ .* β)
    return cgf(layer_, inputs .* β) / β
end

struct StdGaussian{N}
    size::NTuple{N,Int}
    function StdGaussian(n::Int...)
        @assert all(n .≥ 0)
        return new{length(n)}(n)
    end
end

function energy(layer::StdGaussian, x::AbstractArray)
    E = @. x^2 / 2
    return sum_(E; dims = layerdims(layer))
end

function sample_from_inputs(layer::StdGaussian, inputs::AbstractArray, β::Real = one(eltype(inputs)))
    @assert size(inputs) == (size(layer)..., size(inputs)[end])
    x = inputs ./ √β
    z = randn(eltype(x), size(x))
    return @. z / √β + x
end

function cgf(layer::StdGaussian, inputs::AbstractArray, β::Real = one(eltype(inputs)))
    @assert size(inputs) == (size(layer)..., size(inputs)[end])
    Γ = @. inputs^2 / 2 - log(β/π/2)/(2β)
    return sum_(Γ; dims = layerdims(layer))
end

Base.ndims(layer::StdGaussian) = length(layer.size)
Base.size(layer::StdGaussian) = layer.size
Base.length(layer::StdGaussian) = prod(layer.size)
