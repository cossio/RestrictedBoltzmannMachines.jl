struct Spin{T}
    θ::T
end
Spin(::Type{T}, n::Int...) where {T} = Spin(zeros(T, n...))
Spin(n::Int...) = Spin(Float64, n...)
Flux.@functor Spin

function sample_from_inputs(layer::Spin, inputs::AbstractArray)
    x = layer.θ .+ inputs
    pinv = @. one(x) + exp(-2x)
    u = rand(eltype(pinv), size(pinv))
    return @. ifelse(u * pinv ≤ 1, one(x), -one(x))
end

function sample_from_inputs(layer::Spin, inputs::AbstractArray, β::Real)
    layer_ = Spin(layer.θ .* β)
    return sample_from_inputs(layer_, inputs .* β)
end

function cgf(layer::Spin, inputs::AbstractArray)
    x = layer.θ .+ inputs
    Γ = logaddexp.(x, -x)
    return sum_(Γ; dims = layerdims(layer))
end

function cgf(layer::Spin, inputs::AbstractArray, β::Real)
    layer_ = Spin(layer.θ .* β)
    return cgf(layer_, inputs .* β) / β
end

function iterate_states(layer::Spin)
    itr = generate_sequences(length(layer.θ), (-1, 1))
    return map(x -> reshape(x, size(layer.θ)), itr)
end
