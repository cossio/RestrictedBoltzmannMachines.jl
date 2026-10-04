"""
    PottsGumbel(; θ)

Like Potts, but uses the Gumbel-softmax trick for GPU-friendly sampling.
"""
@declare_layer PottsGumbel (θ = zeros,)

# Sampling is the only difference from Potts, whose statistics are shared (see potts.jl).
sample_from_inputs(layer::PottsGumbel, inputs::AbstractArray = Falses(size(layer))) =
    onehot_encode(categorical_sample_from_logits_gumbel(layer.θ .+ inputs), 1:size(layer, 1))
