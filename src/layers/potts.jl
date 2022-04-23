#= As in Binary units, we prefer to encode binary states with
floats 0.0 / 1.0 instead of Booleans, because they hit faster
BLAS linear algebra routines. =#

"""
    Potts(θ)

Potts layer, with external fields `θ`.
Encodes categorical variables as one-hot vectors.
The number of classes is the size of the first dimension.
"""
struct Potts{N, T, A <: AbstractArray{T,N}} <: AbstractLayer{N}
    θ::A
end
Potts(::Type{T}, q::Int, n::Int...) where {T} = Potts(zeros(T, q, n...))
Potts(q::Int, n::Int...) = Potts(Float64, q, n...)

Base.repeat(l::Potts, n::Int...) = Potts(repeat(l.θ, n...))

free_energies(layer::Potts, inputs::Union{Real,AbstractArray} = 0) = -logsumexp(layer.θ .+ inputs; dims=1)
transfer_mean(layer::Potts, inputs::Union{Real,AbstractArray} = 0) = softmax(layer.θ .+ inputs; dims=1)
transfer_mean_abs(layer::Potts, inputs::Union{Real,AbstractArray} = 0) = transfer_mean(layer, inputs)
transfer_std(layer::Potts, inputs::Union{Real,AbstractArray} = 0) = sqrt.(transfer_var(layer, inputs))

function transfer_mode(layer::Potts, inputs::Union{Real,AbstractArray} = 0)
    return layer.θ .+ inputs .== maximum(layer.θ .+ inputs; dims=1)
end

function transfer_var(layer::Potts, inputs::Union{Real,AbstractArray} = 0)
    μ = transfer_mean(layer, inputs)
    return μ .* (1 .- μ)
end

function transfer_meanvar(layer::Potts, inputs::Union{Real,AbstractArray} = 0)
    μ = transfer_mean(layer, inputs)
    ν = μ .* (1 .- μ)
    return μ, ν
end

function transfer_sample(layer::Potts, inputs::Union{Real,AbstractArray} = 0)
    c = categorical_sample_from_logits(layer.θ .+ inputs)
    return onehot_encode(c, 1:colors(layer))
end
