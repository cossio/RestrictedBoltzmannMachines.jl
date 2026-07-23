"""
    Potts(θ)

Layer with Potts units, with external fields `θ`.
Encodes categorical variables as one-hot vectors.
The number of classes is the size of the first dimension.

!!! note
    Sampling from `Potts` layers is not GPU-friendly. For GPU usage,
    use [`PottsGumbel`](@ref) instead, which uses the Gumbel-softmax trick.
"""
@declare_layer Potts (θ = zeros,)

# The statistics are shared with PottsGumbel, which differs from Potts only in how it
# samples (see pottsgumbel.jl).
cgfs(layer::Union{Potts, PottsGumbel}, inputs = 0) = logsumexp(layer.θ .+ inputs; dims = 1)
mean_from_inputs(layer::Union{Potts, PottsGumbel}, inputs = 0) = softmax(layer.θ .+ inputs; dims = 1)
mean_abs_from_inputs(layer::Union{Potts, PottsGumbel}, inputs = 0) = mean_from_inputs(layer, inputs)

function mode_from_inputs(layer::Union{Potts, PottsGumbel}, inputs = 0)
    θ = layer.θ .+ inputs
    return θ .== maximum(θ; dims = 1)
end

function var_from_inputs(layer::Union{Potts, PottsGumbel}, inputs = 0)
    μ = mean_from_inputs(layer, inputs)
    return μ .* (1 .- μ)
end

function meanvar_from_inputs(layer::Union{Potts, PottsGumbel}, inputs = 0)
    μ = mean_from_inputs(layer, inputs)
    ν = μ .* (1 .- μ)
    return μ, ν
end

# Samples are onehot BitArrays, not floats (also for PottsGumbel, which shares this
# encoding). Returning floats here, to avoid the BitArray -> float conversion later in
# inputs_h_from_v / inputs_v_from_h, is not worth it; see the benchmarks in the
# "Design and performance notes" page of the developer docs.
function sample_from_inputs(layer::Potts, inputs = 0)
    c = categorical_sample_from_logits(layer.θ .+ inputs)
    return onehot_encode(c, 1:size(layer, 1))
end
