"""
    ReLU(θ, γ)

ReLU layer, with location parameters `θ` and scale parameters `γ`.
"""
struct ReLU{N, T, A <: AbstractArray{T,N}} <: AbstractLayer{N}
    θ::A
    γ::A
    function ReLU(θ::A, γ::A) where {A<:AbstractArray}
        @assert size(θ) == size(γ)
        return new{ndims(A), eltype(A), A}(θ, γ)
    end
end

ReLU(::Type{T}, n::Int...) where {T} = ReLU(zeros(T, n...), ones(T, n...))
ReLU(n::Int...) = ReLU(Float64, n...)

function effective(layer::ReLU, inputs::AbstractArray; β::Real = true)
    @assert size(layer) == size(inputs)[1:ndims(layer)]
    θ = β * (layer.θ .+ inputs)
    γ = β * broadlike(layer.γ, inputs)
    return ReLU(promote(θ, γ)...)
end

function energies(layer::ReLU, x::AbstractArray)
    @assert size(layer) == size(x)[1:ndims(layer)]
    return relu_energy.(layer.θ, layer.γ, x)
end

free_energies(layer::ReLU) = relu_free.(layer.θ, layer.γ)
transfer_sample(layer::ReLU) = relu_rand.(layer.θ, layer.γ)
transfer_mode(layer::ReLU) = max.(layer.θ ./ abs.(layer.γ), 0)

function transfer_mean(layer::ReLU)
    g = Gaussian(layer.θ, layer.γ)
    μ = transfer_mean(g)
    σ = sqrt.(transfer_var(g))
    return @. μ + σ * tnmean(-μ / σ)
end

transfer_mean_abs(layer::ReLU) = transfer_mean(layer)

function transfer_var(layer::ReLU)
    g = Gaussian(layer.θ, layer.γ)
    μ = transfer_mean(g)
    ν = transfer_var(g)
    return @. ν * tnvar(-μ / √ν)
end

transfer_std(layer::ReLU) = sqrt.(transfer_var(layer))

function ∂free_energy(layer::ReLU)
    μ = transfer_mean(layer)
    ν = transfer_var(layer)
    return (θ = -μ, γ = (ν .+ μ.^2) / 2)
end

function ∂energy(layer::ReLU; xp::AbstractArray, xp2::AbstractArray)
    @assert size(xp) == size(xp2) == size(layer)
    return (; θ = -xp, γ = xp2/2)
end

function sufficient_statistics(layer::ReLU, x::AbstractArray; wts = nothing)
    @assert size(layer) == size(x)[1:ndims(layer)]
    xp = max.(x, 0)
    μp = batchmean(layer, xp; wts)
    μp2 = batchmean(layer, xp.^2; wts)
    return (; xp = μp, xp2 = μp2)
end

function relu_energy(θ::Real, γ::Real, x::Real)
    E = gauss_energy(θ, γ, x)
    if x < 0
        return inf(E)
    else
        return E
    end
end

function relu_free(θ::Real, γ::Real)
    abs_γ = abs(γ)
    return -SpecialFunctions.logerfcx(-θ / √(2abs_γ)) + log(2abs_γ/π) / 2
end

function relu_rand(θ::Real, γ::Real)
    abs_γ = abs(γ)
    μ = θ / abs_γ
    σ = √inv(abs_γ)
    return randnt_half(μ, σ)
end
