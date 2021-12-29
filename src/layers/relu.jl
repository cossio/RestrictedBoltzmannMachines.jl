"""
    ReLU(θ, γ)

ReLU layer, with location parameters `θ` and scale parameters `γ`.
"""
struct ReLU{A<:AbstractArray}
    θ::A
    γ::A
    function ReLU(θ::A, γ::A) where {A<:AbstractArray}
        @assert size(θ) == size(γ)
        return new{A}(θ, γ)
    end
end

ReLU(::Type{T}, n::Int...) where {T} = ReLU(zeros(T, n...), ones(T, n...))
ReLU(n::Int...) = ReLU(Float64, n...)

Flux.@functor ReLU

energies(layer::ReLU, x) = relu_energy.(layer.θ, layer.γ, x)
cgfs(layer::ReLU) = relu_cgf.(layer.θ, layer.γ)
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

function conjugates(layer::ReLU)
    μ = transfer_mean(layer)
    ν = transfer_var(layer)
    return (
        θ = μ,
        γ = @. -(ν + μ^2) / 2
    )
end

function conjugates_empirical(layer::ReLU, samples::AbstractArray)
    @assert size(samples) == (size(layer)..., size(samples)[end])
    xp = max.(samples, 0)
    μ = mean_(xp; dims=ndims(samples))
    μ2 = mean_(xp.^2; dims=ndims(samples))
    return (θ = μ, γ = -μ2/2)
end

function effective(layer::ReLU, inputs, β::Real = true)
    θ = β * (layer.θ .+ inputs)
    γ = β * broadlike(layer.γ, inputs)
    return ReLU(promote(θ, γ)...)
end

function relu_energy(θ::Real, γ::Real, x::Real)
    E = gauss_energy(θ, γ, x)
    if x < 0
        return inf(E)
    else
        return E
    end
end

function relu_cgf(θ::Real, γ::Real)
    abs_γ = abs(γ)
    return SpecialFunctions.logerfcx(-θ / √(2abs_γ)) - log(2abs_γ/π) / 2
end

function relu_rand(θ::Real, γ::Real)
    abs_γ = abs(γ)
    μ = θ / abs_γ
    σ = √inv(abs_γ)
    return randnt_half(μ, σ)
end
