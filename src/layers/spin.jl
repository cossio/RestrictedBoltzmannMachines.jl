@doc raw"""
    Spin(θ)

Spin layer, with external fields `θ`.
The energy of a layer with units ``s_i`` is given by:

```math
E = -\sum_i \theta_i s_i
```

where each spin ``s_i`` takes values ``\pm 1``.
"""
struct Spin{T}
    θ::T
end
Spin(::Type{T}, n::Int...) where {T} = Spin(zeros(T, n...))
Spin(n::Int...) = Spin(Float64, n...)

Flux.@functor Spin

cgfs(layer::Spin) = @. spin_cgf.(layer.θ)

function transfer_sample(layer::Spin)
    u = rand(eltype(layer.θ), size(layer.θ))
    return ifelse.(u .* (1 .+ exp.(-2layer.θ)) .< 1, Int8(1), Int8(-1))
end

transfer_mode(layer::Spin) = ifelse.(layer.θ .> 0, 1, -1)
transfer_mean(layer::Spin) = tanh.(layer.θ)
transfer_mean_abs(layer::Spin) = trues(size(layer))
conjugates(layer::Spin) = (; θ = transfer_mean(layer))
effective(layer::Spin, inputs, β::Real = 1) = Spin(β * (layer.θ .+ inputs))

function transfer_var(layer::Spin)
    μ = transfer_mean(layer)
    return (1 .- μ) .* (1 .+ μ)
end

function spin_cgf(θ::Real)
    abs_θ = abs(θ)
    return abs_θ + LogExpFunctions.log1pexp(-2abs_θ)
end

function conjugates_empirical(layer::Spin, samples::AbstractArray)
    @assert size(samples) == (size(layer)..., size(samples)[end])
    μ = mean_(samples; dims=ndims(samples))
    return (; θ = μ)
end
