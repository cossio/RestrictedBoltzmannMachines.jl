"""
    PottsGumbel(; θ)

Like Potts, but uses the Gumbel-softmax trick for GPU-friendly sampling.
"""
@declare_layer PottsGumbel (θ = zeros,)

# The statistics (`cgfs`, `mean_from_inputs`, ...) are shared with Potts (see potts.jl).
# This is the only change with respect to Potts. Here, we use the Gumbel trick.
function sample_from_inputs(layer::PottsGumbel, inputs = 0)
    c = categorical_sample_from_logits_gumbel(layer.θ .+ inputs)
    return onehot_encode(c, 1:size(layer, 1))
end

function potts_to_gumbel(layer::AbstractLayer)
    if layer isa Potts
        return PottsGumbel(layer)
    else
        return layer
    end
end

function gumbel_to_potts(layer::AbstractLayer)
    if layer isa PottsGumbel
        return Potts(layer)
    else
        return layer
    end
end
