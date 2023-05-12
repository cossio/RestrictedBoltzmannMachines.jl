"""
    Potts(θ)

Layer with Potts units, with external fields `θ`.
Encodes categorical variables as one-hot vectors.
The number of classes is the size of the first dimension.
"""
struct Potts{N,A} <: AbstractLayer{N}
    par::A
    function Potts{N,A}(par::A) where {N,A<:AbstractArray}
        @assert size(par, 1) == 1 # θ
        @assert ndims(par) == N + 1
        return new(par)
    end
end

Potts(par::AbstractArray) = Potts{ndims(par) - 1, typeof(par)}(par)

function Potts(; θ)
    par = vstack((θ,))
    return Potts(par)
end

Potts(::Type{T}, sz::Dims) where {T} = Potts(; θ = zeros(T, sz))
Potts(sz::Dims) = Potts(Float64, sz)

cgfs(layer::Potts, inputs = 0) = logsumexp(layer.θ .+ inputs; dims=1)
mean_from_inputs(layer::Potts, inputs = 0) = softmax(layer.θ .+ inputs; dims=1)
mean_abs_from_inputs(layer::Potts, inputs = 0) = mean_from_inputs(layer, inputs)
std_from_inputs(layer::Potts, inputs = 0) = sqrt.(var_from_inputs(layer, inputs))

function mode_from_inputs(layer::Potts, inputs = 0)
    θ = layer.θ .+ inputs
    return θ .== maximum(θ; dims=1)
end

function var_from_inputs(layer::Potts, inputs = 0)
    μ = mean_from_inputs(layer, inputs)
    return μ .* (1 .- μ)
end

function meanvar_from_inputs(layer::Potts, inputs = 0)
    μ = mean_from_inputs(layer, inputs)
    ν = μ .* (1 .- μ)
    return μ, ν
end

function sample_from_inputs(layer::Potts, inputs = 0)
    c = categorical_sample_from_logits(layer.θ .+ inputs)
    return onehot_encode(c, 1:size(layer, 1))
end
