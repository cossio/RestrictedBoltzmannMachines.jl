#= As in Binary units, we prefer to encode binary states with
floats 0.0 / 1.0 instead of Booleans, because they hit faster
BLAS linear algebra routines. =#

"""
    Potts(θ)

Potts layer, with external fields `θ`.
Encodes categorical variables as one-hot vectors.
The number of classes is the size of the first dimension.
"""
struct Potts{A<:AbstractArray}
    θ::A
end
Potts(::Type{T}, q::Int, n::Int...) where {T} = Potts(zeros(T, q, n...))
Potts(q::Int, n::Int...) = Potts(Float64, q, n...)

Flux.@functor Potts

Base.propertynames(::Potts) = (:q, fieldnames(Potts)...)

function Base.getproperty(layer::Potts, name::Symbol)
    if name == :q
        return size(layer, 1)
    else
        return getfield(layer, name)
    end
end

free_energies(layer::Potts) = -LogExpFunctions.logsumexp(layer.θ; dims = 1)

function transfer_sample(layer::Potts)
    c = categorical_sample_from_logits(layer.θ)
    return onehot_encode(c, 1:layer.q)
end

transfer_mean(layer::Potts) = LogExpFunctions.softmax(layer.θ; dims=1)
transfer_mean_abs(layer::Potts) = transfer_mean(layer)

function transfer_var(layer::Potts)
    p = transfer_mean(layer)
    return p .* (1 .- p)
end

function transfer_mode(layer::Potts)
    return layer.θ .== maximum(layer.θ; dims=1)
end

∂free_energy(layer::Potts) = (; θ = -transfer_mean(layer))
∂energies(::Potts, x::AbstractArray) = (; θ = -x)
effective(layer::Potts, inputs; β::Real = 1) = Potts(β * (layer.θ .+ inputs))
