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

function sample(layer::Spin)
    pinv = @. one(layer.θ) + exp(-2layer.θ)
    u = rand(eltype(pinv), size(pinv))
    return @. ifelse(u * pinv ≤ 1, one(layer.θ), -one(layer.θ))
end

function transform_layer(layer::Spin, inputs, β::Real = 1)
    return Spin(β * (layer.θ .+ inputs))
end

function spin_cgf(θ::Real)
    abs_θ = abs(θ)
    return abs_θ + LogExpFunctions.log1pexp(-2abs_θ)
end
